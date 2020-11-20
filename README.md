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
