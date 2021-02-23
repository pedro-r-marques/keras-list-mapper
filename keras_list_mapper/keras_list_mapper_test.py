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
                assert isinstance(rt, tf.RaggedTensor)
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
                assert isinstance(vals, tf.RaggedTensor)
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
                assert isinstance(vals, tf.RaggedTensor)
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
                assert isinstance(vals, tf.RaggedTensor)
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
