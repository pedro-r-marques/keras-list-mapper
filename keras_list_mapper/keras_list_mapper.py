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
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.python.keras.utils import tf_utils
# NOTE(DavideWalder): is_sequence is not available in the public tf.nest
from tensorflow.python.util import nest
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
        self._output_signatures = None
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
        # Build layer and inner layer
        self.build(nest.map_structure(lambda x: x.shape, inputs))

        inputs_list = tf.nest.flatten(inputs)

        in_functional_construction_mode = any(
            K.is_keras_tensor(t) for t in inputs_list)
        if not (call_context().saving or in_functional_construction_mode):
            return super().__call__(inputs)

        input_spec = nest.map_structure(lambda x: x.type_spec, inputs)
        output_spec = self.compute_output_signature(input_spec)

        if not self._built_from_signature:
            self._output_signatures = output_spec
            self._built_from_signature = True

        outputs = nest.map_structure(self._placeholder_from_spec, output_spec)

        # tracing the function while saving the graph
        if call_context().saving:
            # TODO(DavideWalder): No test coverage
            return nest.map_structure(
                lambda x: x._to_placeholder() if K.is_keras_tensor(x) else x,
                outputs)

        return self._set_connectivity_metadata(
            (inputs,) + args, kwargs, outputs)

    def call(self, inputs, *args, **kwargs):  # pylint: disable=unused-argument
        inputs_list = tf.nest.flatten(inputs)

        input_shape = nest.map_structure(
            lambda x: tf.TensorShape(K.int_shape(x)), inputs)
        batch_inputs = nest.flatten(self._batch_inputs(input_shape))
        first_per_time_inp = next(i for i, batch_inp in enumerate(batch_inputs)
                                  if not batch_inp)

        rt = inputs_list[first_per_time_inp]
        input_rt_shape = rt.bounding_shape()

        sz = rt.row_splits[-1]
        outputs = nest.map_structure(
            lambda x: tf.zeros((sz,) + tuple(x.shape[2:]), dtype=x.dtype),
            self._output_signatures)

        if self.state_shape is not None:
            shp = tf.concat([input_rt_shape[:1], self.state_shape], axis=-1)
            states = tf.zeros(shp)
        else:
            states = tf.zeros((0,))

        row_lengths = rt.row_lengths()
        max_seq_len = input_rt_shape[1]
        for i in tf.range(max_seq_len):
            indices = tf.where(tf.math.less(i, row_lengths))
            indices = tf.cast(indices, row_lengths.dtype)
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

            layer_output = self.layer(inner_inputs, *args, **kwargs)
            if self.state_shape is not None:
                state, *layer_output = layer_output
                states = tf.tensor_scatter_nd_update(states, indices, state)
                if len(layer_output) == 1:
                    layer_output = layer_output[0]

            ix = tf.add(tf.gather_nd(rt.row_splits, indices), xcol)
            ix = tf.expand_dims(ix, 1)

            outputs = nest.map_structure(
                lambda x, y: tf.tensor_scatter_nd_update(x, ix, y),
                outputs, layer_output)

        return nest.map_structure(
            lambda x: tf.RaggedTensor.from_row_splits(x, rt.row_splits),
            outputs)

    @staticmethod
    def _placeholder_from_spec(
            type_spec: Union[tf.TensorSpec, tf.RaggedTensorSpec]):
        return K.placeholder(
            shape=type_spec.shape,
            dtype=type_spec.dtype,
            ragged=isinstance(type_spec, tf.RaggedTensorSpec))

    @staticmethod
    def _remove_timesteps(dims: tf.TensorShape, batch_input: bool
                          ) -> tf.TensorShape:
        if batch_input:
            return dims
        dims = dims.as_list()
        return tf.TensorShape((dims[0], *dims[2:]))

    def _batch_inputs(self, input_shapes) -> Union[bool, List[bool]]:
        batch_inputs: Union[bool, List[bool]] = 0 in self.batch_inputs
        if nest.is_sequence(input_shapes):
            # Check if the input was declared as batch input or if the second
            # dim is not ragged.
            batch_inputs = [i in self.batch_inputs or shape[1] is not None
                            for i, shape in enumerate(input_shapes)]
        return batch_inputs

    def compute_output_shape(self, input_shape):
        input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)

        batch_inputs = self._batch_inputs(input_shape)
        child_input_shape = nest.map_structure(self._remove_timesteps,
                                               input_shape,
                                               batch_inputs)
        if self.state_shape is not None:
            batch_size = nest.flatten(tf_utils.convert_shapes(input_shape))[0]
            state_shape = tf.TensorShape((batch_size, *self.state_shape))
            child_input_shape = [state_shape, *nest.flatten(child_input_shape)]

        child_output_shape = self.layer.compute_output_shape(child_input_shape)

        if self.state_shape is not None:
            child_output_shape = child_output_shape[1:]
            if len(child_output_shape) == 1:
                child_output_shape = child_output_shape[0]
        child_output_shape = tf_utils.convert_shapes(
            child_output_shape, to_tuples=False)
        timesteps = nest.flatten(tf_utils.convert_shapes(input_shape))[1]

        def insert_timesteps(dims: tf.TensorShape):
            dims = dims.as_list()
            return tf.TensorShape([dims[0], timesteps] + dims[1:])

        return nest.map_structure(insert_timesteps, child_output_shape)

    def _create_input_spec(self,
                           spec: Union[tf.TensorSpec, tf.RaggedTensorSpec],
                           shape: tf.TensorShape
                           ) -> Union[tf.TensorSpec, tf.RaggedTensorSpec]:
        if self.mapper_supports_ragged_inputs:
            return tf.RaggedTensorSpec(shape, spec.dtype)
        return tf.TensorSpec(shape, spec.dtype)

    def compute_output_signature(self, input_sig: tf.RaggedTensorSpec):
        input_shape = nest.map_structure(lambda x: x.shape, input_sig)

        batch_inputs = self._batch_inputs(input_shape)
        child_input_shape = nest.map_structure(self._remove_timesteps,
                                               input_shape,
                                               batch_inputs)

        child_input_sig = nest.map_structure(self._create_input_spec,
                                             input_sig,
                                             child_input_shape)
        if self.state_shape is not None:
            batch_size = nest.flatten(tf_utils.convert_shapes(input_shape))[0]
            state_spec = tf.TensorSpec((batch_size, *self.state_shape),
                                       tf.float32)
            child_input_sig = [state_spec, *nest.flatten(child_input_sig)]

        # NOTE(DavideWalder): For many layers, compute_output_signature doesn't
        # give the expected result, therefore the layer is called with
        # placeholders. Alternative:

        # child_output_sig = self.layer.compute_output_signature(
        #     child_input_sig)

        placeholders = nest.map_structure(self._placeholder_from_spec,
                                          child_input_sig)
        child_output_sig = self.layer(placeholders)

        if self.state_shape is not None:
            child_output_sig = child_output_sig[1:]
            if len(child_output_sig) == 1:
                child_output_sig, = child_output_sig
        timesteps = nest.flatten(tf_utils.convert_shapes(input_shape))[1]
        row_splits_dtype = next(
            inp for b_inp, inp in
            zip(*map(nest.flatten, (batch_inputs, input_sig)))
            if not b_inp).row_splits_dtype

        def insert_timesteps(spec):
            shape = spec.shape
            return tf.RaggedTensorSpec([shape[0], timesteps] + shape[1:],
                                       spec.dtype,
                                       row_splits_dtype=row_splits_dtype)
        return nest.map_structure(insert_timesteps, child_output_sig)
