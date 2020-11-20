import unittest

import tensorflow as tf
from tensorflow.keras import layers, models

from .keras_list_mapper import ListMapper


class ListMapperTest(unittest.TestCase):
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
        model.add(layers.Lambda(lambda x: tf.reduce_sum(x)))

        model.compile(loss="mse")

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
        model.add(layers.Lambda(lambda x: tf.reduce_sum(x)))

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
        out = layers.Lambda(lambda x: tf.reduce_sum(x))(mout)
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

        in1 = layers.Input(shape=(None, 2), ragged=True)
        in2 = layers.Input(shape=(None, 3), ragged=True)
        mout = ListMapper(inner)([in1, in2])
        out = layers.Lambda(lambda x: tf.reduce_sum(x))(mout)
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
        model.add(layers.Lambda(lambda x: tf.reduce_sum(x)))

        model.compile(loss="mse")
        y = tf.constant([[1.], [1.]])
        model.fit([rt1], y=y, verbose=0)
