# speak.py --- say something based on the existing model

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # please stfu tensorflow

import tensorflow as tf

one_step_model = tf.saved_model.load('one_step')

states    = None
next_char = tf.constant(["Chomsky says: "])
result    = [next_char]

for n in range(100):
    next_char, states = one_step_model.generate_one_step(
            next_char, states = states
    )
    result.append(next_char)

result = tf.strings.join(result)

print(result[0].numpy().decode('utf-8'))

