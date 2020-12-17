# Keras RaggedTensor Mapper

This package implements a Keras layer that applies a map() operation over
one or more RaggedTensors. This is useful when the application processes
sequences of variables lengths.

For instance, in an NLP context, it is common to process both small and large
documents. For this type of applications RaggedTensors allow the application
to encode the input data as a variable length list (of pages or N paragraphs).

Each of these list elements can then be processed by a neural network that
uses fixed dimension tensors. Often each of these sequence operations wants
to propagate forward state to the next sequence. The ListMapper layer
supports that by allowing the use to define a state vector shape.

## Example

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras_list_mapper.keras_list_mapper import ListMapper


class RecurrentCell(layers.Layer):
    """ Example recurrent cell
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        state, features = inputs
        nstate = state + features
        output = tf.reduce_mean(nstate, axis=-1)
        return nstate, output


def make_model():
    inp = layers.Input(shape=(None, 4), ragged=True)
    map_fn = RecurrentCell()
    m = ListMapper(map_fn, state_shape=(4,))
    mr = m(inp)
    s = layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1))(mr)
    model = keras.Model(inp, s)
    model.compile(optimizer="adam", loss="mse")
    return model

model = make_model()

values = tf.reshape(tf.range(32), (8, 4))
x = tf.RaggedTensor.from_row_lengths(values, [3, 2, 2, 1])
model.predict(x)
```

In this example a RecurrentCell is applied over the ragged dimension of the tensor. The current cell performs a computation and stores state in a state vector.

The Ragged Tensor ```x``` has a shape of [4, None, 4]; 4 batches having a sequence length of [3, 2, 2, 1] and then a feature dimension of 4.

The function of the ListMapper is to call the RecurrentCell for the valid
(batch, sequence) pairs, providing an additional state vector per batch.
