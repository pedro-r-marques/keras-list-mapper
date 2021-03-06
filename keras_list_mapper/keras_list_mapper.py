# Copyright 2020 Pedro R. Marques. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Keras RaggedTensor ListMapper layer
"""
from typing import Any, Dict, List, Optional

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.python.util.keras_deps import get_call_context_function


def call_context():
    fn = get_call_context_function()
    return fn()


class ListMapper(layers.Wrapper):
    """Apply a mapper function to one or more RaggedTensor(s).

    This layer applies a map operation over a RaggedTensor sequence
    dimension. It calls the mapper operation for each sequence grouping
    together the batch indicies for which the sequence is valid.

    It supports an optional state vector which can be an input of the mapper
    and updated at every step.

    Arguments:
      layer: mapper layer to call for each RaggedTensor sequence.
      state_shape: TensorShape for state vector.
      batch_inputs: By default, ragged tensors passed to the mapper node
        split by time_step dimension. This option overrides this behavior
        allowing for a given input to be passed through unmodified for all
        time-steps of the batch.

    The mapper layer should accept as input shape the same shapes as this
    layer with the exception that the sequence_len dimension of RaggedTensors
    is removed. In addition, when a state_shape is specified, this layer
    expect the mapper to accept as first input the state vector and return
    the state also as the first output tensor.

    Input shape:
      Zero or more N-D Tensor(s) with shape:
      `(batch_size, ...)`
      One or more N-D RaggedTensor(s) with shape:
      `(batch_size, (sequence_length), features...)`.

    Output shape:
      `(batch_size, (sequence_length), mapper_dims...).

    """

    def __init__(self, layer: layers.Layer,
                 state_shape: Optional[tf.TensorShape] = None,
                 batch_inputs: Optional[List[int]] = None,
                 mapper_supports_ragged_inputs: Optional[bool] = None,
                 **kwargs):
        super().__init__(layer, **kwargs)
        if state_shape is not None and \
                not isinstance(state_shape, tf.TensorShape):
            state_shape = tf.TensorShape(state_shape)
        self.state_shape = state_shape
        if batch_inputs is None:
            self.batch_inputs = set()
        else:
            self.batch_inputs = set(batch_inputs)
        self._built_from_signature = False
        self._output_signature = None
        if mapper_supports_ragged_inputs is None:
            ragged_in = getattr(self.layer, '_supports_ragged_inputs', False)
        else:
            ragged_in = mapper_supports_ragged_inputs
        self.mapper_supports_ragged_inputs = ragged_in

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "state_shape": self.state_shape,
            "batch_inputs": list(self.batch_inputs),
            "mapper_supports_ragged_inputs": self.mapper_supports_ragged_inputs
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects=None):
        config["batch_inputs"] = set(config["batch_inputs"])
        return super().from_config(config, custom_objects)

    def __call__(self, inputs, *args, **kwargs):
        inputs_list = tf.nest.flatten(inputs)

        in_functional_construction_mode = any(
            K.is_keras_tensor(t) for t in inputs_list)
        if not (call_context().saving or in_functional_construction_mode):
            return super().__call__(inputs)

        inner_inputs = []
        if self.state_shape is not None:
            inner_inputs.append(
                K.placeholder(shape=[None] + self.state_shape))

        count = 0
        for ix, inp in enumerate(inputs_list):
            # TODO(roque): remove _type_spec after v2.3
            spec = getattr(inp, 'type_spec', None)
            if spec is None:
                spec = getattr(inp, '_type_spec', None)
            if spec is None or not isinstance(spec, tf.RaggedTensorSpec) \
                    or ix in self.batch_inputs:
                inner_inputs.append(inp)
                continue
            count += 1
            # TODO(roque): remove _shape after v2.3
            if hasattr(spec, 'shape'):
                nshape = tf.TensorShape(spec.shape[:1] + spec.shape[2:])
            else:
                nshape = tf.TensorShape(spec._shape[:1] + spec._shape[2:])
            assert isinstance(spec, tf.RaggedTensorSpec)
            is_ragged = self.mapper_supports_ragged_inputs
            pl = K.placeholder(shape=nshape, dtype=inp.dtype, ragged=is_ragged)
            inner_inputs.append(pl)

        if not count:
            raise ValueError(
                'ListMapper inputs has no RaggedTensor')

        if len(inner_inputs) == 1:
            inner_inputs = inner_inputs[0]
        outputs = self.layer(inner_inputs, *args, **kwargs)

        # tracing the function while saving the graph
        if call_context().saving:
            if self.state_shape is not None:
                outputs = outputs[1:]
            elif not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            outputs = [x._to_placeholder() if K.is_keras_tensor(x) else x
                       for x in outputs]
            return outputs

        if not self._built_from_signature:
            if self.state_shape is not None:
                outputs = outputs[1]
            # TODO(roque): remove _type_spec
            self._output_signature = getattr(
                outputs, 'type_spec', tf.TensorSpec(outputs.shape))
            self._built_from_signature = True

        super().build(inputs)
        self._set_connectivity_metadata((inputs,) + args, kwargs, outputs)
        return outputs

    def call(self, inputs, *args, **kwargs):  # pylint: disable=unused-argument
        inputs_list = tf.nest.flatten(inputs)

        per_time_step_inputs = []
        for ix, inp in enumerate(inputs_list):
            if ix in self.batch_inputs:
                continue
            if isinstance(inp, tf.RaggedTensor):
                per_time_step_inputs.append(ix)

        rt = inputs_list[per_time_step_inputs[0]]

        input_rt_shape = rt.bounding_shape()

        sz = rt.row_splits[-1]
        outputs = tf.zeros((sz,) + tuple(self._output_signature.shape[1:]))

        if self.state_shape is not None:
            shp = tf.concat([input_rt_shape[:1], self.state_shape], axis=-1)
            states = tf.zeros(shp)
        else:
            states = tf.zeros((0,))

        row_lengths = rt.row_lengths()
        max_seq_len = input_rt_shape[1]
        for i in tf.range(max_seq_len):
            indices = tf.where(tf.math.less(i, row_lengths))
            xcol = tf.fill([tf.shape(indices)[0]], i)
            col_indices = tf.concat([indices, tf.expand_dims(xcol, 1)], axis=1)

            inner_inputs = []
            if self.state_shape is not None:
                state = tf.gather_nd(states, indices)
                inner_inputs.append(state)

            for ix, inp in enumerate(inputs_list):
                if ix not in self.batch_inputs and \
                        isinstance(inp, tf.RaggedTensor):
                    x = tf.gather_nd(inp, col_indices)
                    if not self.mapper_supports_ragged_inputs and isinstance(
                            x, tf.RaggedTensor):
                        x = x.to_tensor()
                else:
                    x = tf.gather_nd(inp, indices)
                inner_inputs.append(x)

            if len(inner_inputs) == 1:
                inner_inputs = inner_inputs[0]

            y = self.layer(inner_inputs, *args, **kwargs)
            if self.state_shape is not None:
                state, output = y
                states = tf.tensor_scatter_nd_update(states, indices, state)
            else:
                output = y

            ix = tf.add(tf.gather_nd(rt.row_splits, indices), xcol)
            ix = tf.expand_dims(ix, 1)

            outputs = tf.tensor_scatter_nd_update(outputs, ix, output)

        return tf.RaggedTensor.from_row_splits(outputs, rt.row_splits)
