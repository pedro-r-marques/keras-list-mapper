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
""" Keras ListMapper unit tests.
"""
import itertools
import tempfile
import unittest
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models

from .keras_list_mapper import ListMapper


class ListMapperTest(unittest.TestCase):
    """ ListMapper unittests.
    """

    def test_1rt_arg(self):
        """ inner layer with 1 ragged tensor argument
        """
        inner = layers.Dense(1)

        rt1 = tf.ragged.constant([[1., 2.], [3., .4], [5., 6.]])
        rt2 = tf.ragged.constant([[10., 20.], [30., .40]])
        rt = tf.ragged.stack([rt1, rt2])

        model = models.Sequential()
        model.add(layers.Input(shape=(None, 2), ragged=True))
        model.add(ListMapper(inner))
        model.add(layers.Lambda(tf.reduce_sum))

        model.compile(loss="mse")
        model.summary(print_fn=lambda x: None)

        y = tf.constant([[19.], [100.]])
        model.fit(x=rt, y=y, verbose=0)

    def test_eager(self):
        """ inner layer with 1 ragged tensor argument
        """
        inner = layers.Dense(1)

        rt1 = tf.ragged.constant([[1., 2.], [3., .4], [5., 6.]])
        rt2 = tf.ragged.constant([[10., 20.], [30., .40]])
        rt = tf.ragged.stack([rt1, rt2])

        model = models.Sequential()
        model.add(layers.Input(shape=(None, 2), ragged=True))
        model.add(ListMapper(inner))
        model.add(layers.Lambda(tf.reduce_sum))

        model.compile(loss="mse", run_eagerly=True)

        y = tf.constant([[19.], [100.]])
        model.fit(x=rt, y=y, verbose=0)

    def test_2rt_arg(self):
        """ inner layer with 2 ragged tensor arguments
        """
        inner_in1 = layers.Input(shape=(2,))
        inner_in2 = layers.Input(shape=(3,))
        inner_vec = layers.Concatenate()([inner_in1, inner_in2])
        inner_out = layers.Dense(1)(inner_vec)
        inner = models.Model([inner_in1, inner_in2], inner_out)

        c11 = tf.ragged.constant([[1., 2.], [3., .4], [5., 6.]])
        c12 = tf.ragged.constant([[10., 20.], [30., .40]])
        rt1 = tf.ragged.stack([c11, c12])

        c21 = tf.ragged.constant([[1., 2., 3.], [.4, 5., 6.], [7., 8., 9.]])
        c22 = tf.ragged.constant([[10., 20., 30.], [.40, 50., 60.]])
        rt2 = tf.ragged.stack([c21, c22])

        in1 = layers.Input(shape=(None, 2), ragged=True)
        in2 = layers.Input(shape=(None, 3), ragged=True)
        mout = ListMapper(inner)([in1, in2])
        out = layers.Lambda(tf.reduce_sum)(mout)
        model = models.Model([in1, in2], out)

        model.compile(loss="mse")

        y = tf.constant([[1.], [1.]])
        model.fit([rt1, rt2], y=y, verbose=0)

    def test_mixed_tensors(self):
        """ inner layer with array tensor and ragged tensor
        """
        inner_in1 = layers.Input(shape=(2,))
        inner_in2 = layers.Input(shape=(3,))
        inner_vec = layers.Concatenate()([inner_in1, inner_in2])
        inner_out = layers.Dense(1)(inner_vec)
        inner = models.Model([inner_in1, inner_in2], inner_out)

        rt1 = tf.constant([[1., 2], [3., 4]])

        c21 = tf.ragged.constant([[1., 2., 3.], [.4, 5., 6.], [7., 8., 9.]])
        c22 = tf.ragged.constant([[10., 20., 30.], [.40, 50., 60.]])
        rt2 = tf.ragged.stack([c21, c22])

        in1 = layers.Input(shape=(2,))
        in2 = layers.Input(shape=(None, 3), ragged=True)
        mout = ListMapper(inner)([in1, in2])
        out = layers.Lambda(tf.reduce_sum)(mout)
        model = models.Model([in1, in2], out)

        model.compile(loss="mse")

        y = tf.constant([[1.], [1.]])
        model.fit([rt1, rt2], y=y, verbose=0)

    def test_state(self):
        """ inner layer with state and 1 ragged tensor
        """
        inner_in1 = layers.Input(shape=(3,))
        inner_in2 = layers.Input(shape=(3,))
        sum_vec = layers.Add()([inner_in1, inner_in2])

        def xsum(x):
            return tf.reduce_sum(x, axis=1, keepdims=True)
        out = layers.Lambda(xsum)(sum_vec)
        inner = models.Model([inner_in1, inner_in2], [sum_vec, out])

        c1 = tf.ragged.constant([[1., 2., 3.], [.4, 5., 6.], [7., 8., 9.]])
        c2 = tf.ragged.constant([[10., 20., 30.], [.40, 50., 60.]])
        rt1 = tf.ragged.stack([c1, c2])

        model = models.Sequential()
        model.add(layers.Input(shape=(None, 3), ragged=True))
        model.add(ListMapper(inner, state_shape=tf.TensorShape(3,)))
        model.add(layers.Lambda(tf.reduce_sum))

        model.compile(loss="mse")
        y = tf.constant([[1.], [1.]])
        model.fit([rt1], y=y, verbose=0)

    def test_training_flag(self):
        """ ensure that the training flag is correctly propagated
        """
        class FakeLayer(layers.Layer):
            def call(self, inputs, **kwargs):
                x = tf.ones_like(inputs)
                y = tf.zeros_like(inputs)
                return K.in_train_phase(
                    x, y, training=kwargs.get('training', None))

        model = models.Sequential()
        model.add(layers.Input(shape=(None, 2), ragged=True))
        model.add(ListMapper(FakeLayer()))

        def reduce_sum(x):
            # input shape [batch, (sequence), 2]
            # reduce to a sum of values per batch.
            s = tf.ragged.map_flat_values(tf.reduce_sum, x, axis=[-1])
            if isinstance(s, tf.RaggedTensor):
                s = s.to_tensor()
                return tf.reduce_sum(s, axis=-1)
            return s
        model.add(layers.Lambda(reduce_sum))

        model.compile(loss="mae")

        rt1 = tf.ragged.constant([[1., 1.]])
        rt2 = tf.ragged.constant([[1., 1.], [1., .1]])
        # tf.ragged.stack requires axis=0 to preserve order
        rt = tf.ragged.stack([rt1, rt2], axis=0)
        y = tf.constant([[2.], [4.]])
        history = model.fit(x=rt, y=y, verbose=0)
        loss = history.history['loss'][-1]
        self.assertEqual(loss, 0.0)

    def test_batch_inputs(self):
        """ Test the batch_inputs parameter

        Batch inputs are not processed on a per time step basis
        """
        rt_shape = []

        class FakeLayer(layers.Layer):
            def call(self, inputs, *kwargs):
                state, rt, vals = inputs
                rt_shape.append(rt.shape)
                x = tf.reduce_sum(rt, axis=[-1])
                x = tf.expand_dims(x, -1)
                state = tf.math.add(state, vals)
                y = tf.math.add(state, x)
                return state, y

        in_batch = layers.Input(shape=(None,), ragged=True)
        in_ts = layers.Input(shape=(None, 2), ragged=True)
        lm = ListMapper(FakeLayer(), state_shape=(2,), batch_inputs=[0])
        out = lm([in_batch, in_ts])

        model = models.Model([in_batch, in_ts], out)
        model.compile(loss="mae")

        rt_batch = tf.ragged.constant([[1, 2, 0, 3], [0, 1, 2]])
        rt_ts = tf.ragged.constant([
            [[1, 2], [3, 4], [1, 0]],
            [[1, 2]]
        ])
        y = model.predict([rt_batch, rt_ts])

        rank = {len(x) for x in rt_shape}
        self.assertEqual(list(rank), [2])
        self.assertEqual(y[0].numpy().tolist(), [
            [7.0, 8.0], [10.0, 12.0], [11.0, 12.0]
        ])
        self.assertEqual(y[1].numpy().tolist(), [[4.0, 5.0]])

    def test_ragged_placeholder(self):
        class FakeLayer(layers.Layer):
            def call(self, inputs, *kwargs):
                state, vals = inputs
                x = tf.reduce_sum(vals, axis=1)
                if isinstance(x, tf.RaggedTensor):
                    x = x.to_tensor()
                state = tf.math.add(state, x)
                return state, state

        model = models.Sequential()
        model.add(layers.Input(shape=(None, None, 4), ragged=True))
        fake = FakeLayer()
        model.add(
            ListMapper(fake, state_shape=(4,),
                       mapper_supports_ragged_inputs=True))

        model.compile()

        rt = tf.ragged.stack([
            tf.ragged.constant(np.arange(24).reshape(2, 3, 4)),
            tf.ragged.constant(np.arange(60).reshape(3, 5, 4)),
        ])
        model.predict(rt)

    def test_save(self):
        class FakeLayer(layers.Layer):
            def call(self, inputs, *kwargs):
                state, vals = inputs
                x = tf.reduce_sum(vals, axis=1)
                if isinstance(x, tf.RaggedTensor):
                    x = x.to_tensor()
                state = tf.math.add(state, x)
                return state, state

        model = models.Sequential()
        model.add(layers.Input(shape=(None, None, 4), ragged=True))
        fake = FakeLayer()
        model.add(
            ListMapper(fake, state_shape=(4,),
                       mapper_supports_ragged_inputs=True))
        model.compile(loss="mse")

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir, save_traces=False)

    def test_save_to_json(self):
        """ Test that ListMapper correctly serializes to JSON """

        class FakeLayer(layers.Layer):
            def call(self, inputs, *kwargs):
                state, vals = inputs
                x = tf.reduce_sum(vals, axis=1)
                if isinstance(x, tf.RaggedTensor):
                    x = x.to_tensor()
                state = tf.math.add(state, x)
                return state, state

        model = models.Sequential()
        model.add(layers.Input(shape=(None, None, 4), ragged=True))
        fake = FakeLayer(name="fake_layer")
        model.add(
            ListMapper(fake, state_shape=(4,),
                       mapper_supports_ragged_inputs=True,
                       name="list_mapper"))
        model.compile(loss="mse")

        model_json = model.to_json()
        from_json_model = models.model_from_json(
            model_json,
            custom_objects={"ListMapper": ListMapper, "FakeLayer": FakeLayer})

        # Test same structure
        for layer, new_layer in itertools.zip_longest(model.layers,
                                                      from_json_model.layers):
            self.assertEqual(layer.get_config(), new_layer.get_config())

        # Test same output
        rt = tf.ragged.stack([
            tf.ragged.constant(np.arange(24).reshape(2, 3, 4)),
            tf.ragged.constant(np.arange(60).reshape(3, 5, 4)),
        ])
        tf.assert_equal(model.predict(rt).to_tensor(),
                        from_json_model.predict(rt).to_tensor())

    def test_multioutput_layer(self):
        """
        Tests that ListMapper correctly handles inner layers with multiple
        outputs.
        """
        inner_in1 = layers.Input(shape=(2,))
        inner_in2 = layers.Input(shape=(3,))
        inner_vec = layers.Concatenate()([inner_in1, inner_in2])
        inner_out1 = layers.Dense(1)(inner_vec)
        inner_out2 = layers.Dense(2)(inner_vec)
        inner = models.Model([inner_in1, inner_in2], [inner_out1, inner_out2])

        rt1 = tf.constant([[1., 2], [3., 4]])

        c21 = tf.ragged.constant([[1., 2., 3.], [.4, 5., 6.], [7., 8., 9.]])
        c22 = tf.ragged.constant([[10., 20., 30.], [.40, 50., 60.]])
        rt2 = tf.ragged.stack([c21, c22])

        in1 = layers.Input(shape=(2,))
        in2 = layers.Input(shape=(None, 3), ragged=True)
        mout1, mout2 = ListMapper(inner, batch_inputs=[0])([in1, in2])
        out1 = layers.Lambda(tf.reduce_sum)(mout1)
        out2 = layers.Lambda(tf.reduce_sum)(mout2)
        out = layers.Add()([out1, out2])
        model = models.Model([in1, in2], out)

        model.compile(loss="mse")

        y = tf.constant([[1.], [1.]])
        model.fit([rt1, rt2], y=y, verbose=0)

    def test_dtypes(self):
        """
        Test that dtypes are correct during graph construction and prediction.
        """

        inner_in = layers.Input(shape=(3,))
        inner_out1 = layers.Lambda(lambda x: tf.cast(x, tf.bool))(inner_in)
        inner_out2 = layers.Lambda(lambda x: tf.cast(x, tf.int32))(inner_in)
        inner = models.Model(inner_in, [inner_out1, inner_out2])

        inp = layers.Input(shape=(None, 3), ragged=True)
        mout1, mout2 = ListMapper(inner)([inp])
        self.assertEqual(mout1.dtype, inner.output[0].dtype)
        self.assertEqual(mout2.dtype, inner.output[1].dtype)
        model = models.Model([inp], [mout1, mout2])

        model.compile(loss="mse")

        c21 = tf.ragged.constant([[1., 2., 3.], [.4, 5., 6.], [7., 8., 9.]])
        c22 = tf.ragged.constant([[10., 20., 30.], [.40, 50., 60.]])
        rt2 = tf.ragged.stack([c21, c22])

        pred = model.predict([rt2])
        self.assertEqual(pred[0].dtype, inner.output[0].dtype)
        self.assertEqual(pred[1].dtype, inner.output[1].dtype)

    def test_multioutput_layer_with_state(self):
        """
        Tests that ListMapper correctly handles multi-input,
        multi-outputs inner layers and state.
        """

        state = layers.Input(shape=(1,))
        inner_in1 = layers.Input(shape=(2,))
        inner_in2 = layers.Input(shape=(3,))
        inner_vec = layers.Concatenate()([state, inner_in1, inner_in2])
        inner_out1 = layers.Dense(1)(inner_vec)
        inner_out2 = layers.Dense(2)(inner_vec)
        inner = models.Model([state, inner_in1, inner_in2],
                             [state, inner_out1, inner_out2])

        rt1 = tf.constant([[1., 2], [3., 4]])

        c21 = tf.ragged.constant([[1., 2., 3.], [.4, 5., 6.], [7., 8., 9.]])
        c22 = tf.ragged.constant([[10., 20., 30.], [.40, 50., 60.]])
        rt2 = tf.ragged.stack([c21, c22])

        in1 = layers.Input(shape=(2,))
        in2 = layers.Input(shape=(None, 3), ragged=True)
        mout1, mout2 = ListMapper(
            inner, batch_inputs=[0], state_shape=(1,))([in1, in2])
        out1 = layers.Lambda(tf.reduce_sum)(mout1)
        out2 = layers.Lambda(tf.reduce_sum)(mout2)
        out = layers.Add()([out1, out2])
        model = models.Model([in1, in2], out)

        model.compile(loss="mse")

        y = tf.constant([[1.], [1.]])
        model.fit([rt1, rt2], y=y, verbose=0)

    def test_lstm_after_list_mapper(self):
        """ Test that ListMapper can be used as an intermediate layer.
        """

        inner_in1 = layers.Input(shape=(2,))
        inner_in2 = layers.Input(shape=(3,))
        inner_vec = layers.Concatenate()([inner_in1, inner_in2])
        inner_out = layers.Dense(1)(inner_vec)
        inner = models.Model([inner_in1, inner_in2], inner_out)

        rt1 = tf.constant([[1., 2], [3., 4]])

        c21 = tf.ragged.constant([[1., 2., 3.], [.4, 5., 6.], [7., 8., 9.]])
        c22 = tf.ragged.constant([[10., 20., 30.], [.40, 50., 60.]])
        rt2 = tf.ragged.stack([c21, c22])

        in1 = layers.Input(shape=(2,))
        in2 = layers.Input(shape=(None, 3), ragged=True)
        mout = ListMapper(inner, batch_inputs=[0])([in1, in2])
        lstm = layers.LSTM(4, return_sequences=True)(mout)
        td = layers.TimeDistributed(layers.Dense(2))(lstm)
        out = layers.Lambda(tf.reduce_sum)(td)
        model = models.Model([in1, in2], out)

        model.compile(loss="mse")

        y = tf.constant([[1.], [1.]])
        model.fit([rt1, rt2], y=y, verbose=0)

    def test_row_split_dtype(self):
        """
        Test the the output's row_splits dtype is inherited from the first
        input.
        """

        test_dtypes = [
            (tf.int32, tf.int32),
            (tf.int32, tf.int64),
            (tf.int64, tf.int32),
            (tf.int64, tf.int64),
        ]

        inner_in1 = layers.Input(shape=(2,))
        inner_in2 = layers.Input(shape=(3,))
        inner_vec = layers.Concatenate()([inner_in1, inner_in2])
        inner = models.Model([inner_in1, inner_in2], inner_vec)

        in1 = layers.Input(shape=(None, 2), ragged=True)
        in2 = layers.Input(shape=(None, 3), ragged=True)
        mout = ListMapper(inner)([in1, in2])
        model = models.Model([in1, in2], mout)

        model.compile(loss="mse")

        c11 = tf.ragged.constant([[1., 2.], [3., .4], [5., 6.]])
        c12 = tf.ragged.constant([[10., 20.], [30., .40]])
        rt1 = tf.ragged.stack([c11, c12])

        c21 = tf.ragged.constant([[1., 2., 3.], [.4, 5., 6.], [7., 8., 9.]])
        c22 = tf.ragged.constant([[10., 20., 30.], [.40, 50., 60.]])
        rt2 = tf.ragged.stack([c21, c22])

        for rt1_dtype, rt2_dtype in test_dtypes:
            with self.subTest(rt1_dtype=rt1_dtype, rt2_dtype=rt2_dtype):
                rt1 = rt1.with_row_splits_dtype(rt1_dtype)
                rt2 = rt2.with_row_splits_dtype(rt2_dtype)
                out_rt = model.predict([rt1, rt2])
                self.assertEqual(out_rt.row_splits.dtype, rt1_dtype)


class ComputeOutputShapeTest(unittest.TestCase):
    """ Test output shape inference. """

    def _runner(self, mapper: ListMapper, test_shapes: List[Tuple]):
        for inp_shape, exp_shape in test_shapes:
            with self.subTest(inp_shape=inp_shape, exp_shape=exp_shape):
                out_shape = mapper.compute_output_shape(inp_shape)
                if isinstance(exp_shape[0], list):
                    self.assertEqual(exp_shape,
                                     [shape.as_list() for shape in out_shape])
                else:
                    self.assertEqual(exp_shape, out_shape.as_list())

    def test_single_input_no_state(self):

        test_shapes = [
            ([None, None, 4], [None, None, 3]),
            ([5, 12, 1], [5, 12, 3]),
        ]

        mapper = ListMapper(layers.Dense(3))
        self._runner(mapper, test_shapes)

    def test_single_input_with_state(self):
        test_shapes = [
            ([None, None, 5], [None, None, 7]),
            ([5, 12, 5], [5, 12, 7]),
        ]

        state_inp = layers.Input(shape=(2,))
        inner_in = layers.Input(shape=(5,))
        inner_vec = layers.Concatenate()([state_inp, inner_in])
        inner = models.Model([state_inp, inner_in],
                             [state_inp, inner_vec])

        mapper = ListMapper(inner, state_shape=(2,))
        self._runner(mapper, test_shapes)

    def test_multiple_input_with_state(self):
        test_shapes = [
            ([[None, None, 5], [None, None, 7, 11]], [None, None, 79]),
            # The second input will be used as time step input since it's not
            # ragged in the second dim.
            ([[1, None, 5], [1, 2, 7, 11]], [1, None, 156]),
        ]

        state_inp = layers.Input(shape=(2,))
        inner_in0 = layers.Input(shape=(5,))
        inner_in1 = layers.Input(shape=(7, 11))
        inner_in1_f = layers.Flatten()(inner_in1)
        inner_vec = layers.Concatenate()([state_inp, inner_in1_f])
        inner = models.Model([state_inp, inner_in0, inner_in1],
                             [state_inp, inner_vec])

        mapper = ListMapper(inner, state_shape=(2,))
        self._runner(mapper, test_shapes)

    def test_multiple_input_multiple_output_with_state(self):
        test_shapes = [
            ([[None, None, 5], [None, None, 7, 11]],
             [[None, None, 79], [None, None]]),

            # The second input will be used as time step input since it's not
            # ragged in the second dim.
            ([[1, None, 5], [1, 2, 7, 11]],
             [[1, None, 156], [1, None]]),
        ]

        state_inp = layers.Input(shape=(2,))
        inner_in0 = layers.Input(shape=(5,))
        inner_in1 = layers.Input(shape=(7, 11))
        inner_in1_f = layers.Flatten()(inner_in1)
        inner_vec = layers.Concatenate()([state_inp, inner_in1_f])
        inner_dense = layers.Reshape(tuple())(layers.Dense(1)(inner_vec))
        inner = models.Model([state_inp, inner_in0, inner_in1],
                             [state_inp, inner_vec, inner_dense])

        mapper = ListMapper(inner, state_shape=(2,))
        self._runner(mapper, test_shapes)

    def test_single_input_recursive(self):
        test_shapes = [
            ([None, None, None, 4], [None, None, None, 3]),
        ]

        # TODO(DavideWalder): Add full recursive usage support
        mapper = ListMapper(ListMapper(layers.Dense(3)))
        self._runner(mapper, test_shapes)


class ComputeOutputSignatureTest(unittest.TestCase):
    """ Test output signature inference

    Test coverage of compute_output_signature is also provided by
    ListMapperTest since it used in __call__.
    """

    def _runner(self, mapper: ListMapper, test_sigs: List[Tuple]):
        for inp_sig, exp_sig in test_sigs:
            with self.subTest(inp_shape=inp_sig, exp_shape=exp_sig):
                out_sig = mapper.compute_output_signature(inp_sig)
                self.assertEqual(exp_sig, out_sig)

    def test_single_input_no_state(self):
        test_specs = [
            (tf.RaggedTensorSpec([None, None, 4], tf.float32, row_splits_dtype=tf.int32),  # noqa E501
             tf.RaggedTensorSpec([None, None, 3], tf.int64, row_splits_dtype=tf.int32)),  # noqa E501

            (tf.RaggedTensorSpec([5, 2, 4], tf.float32, row_splits_dtype=tf.int32),  # noqa E501
             tf.RaggedTensorSpec([5, 2, 3], tf.int64, row_splits_dtype=tf.int32)),  # noqa E501
        ]

        inner_in = layers.Input(shape=(4,), dtype=tf.float32)
        inner_dense = layers.Dense(3)
        inner_cast = layers.Lambda(lambda x: tf.cast(x, tf.int64))
        inner = tf.keras.Sequential([inner_in, inner_dense, inner_cast])
        mapper = ListMapper(inner)
        self._runner(mapper, test_specs)

    def test_multiple_input_multiple_output_with_state(self):
        test_specs = [
            # Test that row_splits_dtype comes from the first input
            ([tf.RaggedTensorSpec([None, None, 5], tf.float32, row_splits_dtype=tf.int64),  # noqa E501
              tf.RaggedTensorSpec([None, None, 7, 11], tf.float32, row_splits_dtype=tf.int32)],  # noqa E501
             [tf.RaggedTensorSpec([None, None, 79], tf.float32, row_splits_dtype=tf.int64),  # noqa E501
              tf.RaggedTensorSpec([None, None], tf.float32, row_splits_dtype=tf.int64)]),  # noqa E501
        ]

        state_inp = layers.Input(shape=(2,))
        inner_in0 = layers.Input(shape=(5,))
        inner_in1 = layers.Input(shape=(7, 11))
        inner_in1_f = layers.Flatten()(inner_in1)
        inner_vec = layers.Concatenate()([state_inp, inner_in1_f])
        inner_dense = layers.Reshape(tuple())(layers.Dense(1)(inner_vec))
        inner = models.Model([state_inp, inner_in0, inner_in1],
                             [state_inp, inner_vec, inner_dense])

        mapper = ListMapper(inner, state_shape=(2,))
        self._runner(mapper, test_specs)
