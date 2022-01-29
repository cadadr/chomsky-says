# model.py --- the model

import tensorflow as tf

class ChomskySaysModel(tf.keras.Model):

    def __init__(self, vocabulary_size, embedding_dimension, num_rnn_units):

        super().__init__(self)

        self.embedding = tf.keras.layers.Embedding(
                vocabulary_size, embedding_dimension
        )

        self.GRU = tf.keras.layers.GRU(
                num_rnn_units,
                return_sequences = True,
                return_state     = True
        )

        self.dense = tf.keras.layers.Dense(vocabulary_size)


    def call(self, inputs, states = None, return_state = False,
            training = False):

        x = self.embedding(inputs, training = training)

        if states is None:
            states = self.GRU.get_initial_state(x)

        x, states = self.GRU(
                x, initial_state = states, training = training
        )

        x = self.dense(x, training = training)

        if return_state:
            return x, states

        else:
            return x


