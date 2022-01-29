# onestep.py --- single step generator

import tensorflow as tf

class OneStep(tf.keras.Model):

    def __init__(self, model, chars_from_ids, ids_from_chars,
            temperature = 1.0):

        super().__init__()

        self.temperature = temperature
        self.model       = model
        self.c4i         = chars_from_ids
        self.i4c         = ids_from_chars

        # Create a mask to prevent [UNK]
        skip_ids    = self.i4c(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
                # Put -inf at bad indices
                values      = [ -float('inf') ] * len(skip_ids),
                indices     = skip_ids,
                dense_shape = [len(self.i4c.get_vocabulary())]
        )

        self.prediction_mask = tf.sparse.to_dense(sparse_mask)


    @tf.function
    def generate_one_step(self, inputs, states = None):

        in_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        in_ids   = self.i4c(in_chars).to_tensor()

        predicted_logits, states = \
            self.model(
                    inputs = in_ids, states = states, return_state = True
            )

        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = \
            (predicted_logits / self.temperature) + self.prediction_mask

        predicted_ids = tf.random.categorical(
                predicted_logits, num_samples = 1
        )
        predicted_ids = tf.squeeze(predicted_ids, axis = -1)

        predicted_chars = self.c4i(predicted_ids)

        return predicted_chars, states

