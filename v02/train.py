# v02/train.py --- train the model

import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # please stfu tensorflow

import numpy as np
import tensorflow as tf

from model   import ChomskySaysModel
from onestep import OneStep
from param   import *

# Setup

corpus      = open(corpus_path, 'r').read()
corpus_size = len(corpus)

if corpus_sliced:
    corpus_mid = corpus_size // 2
    corpus     = corpus[ corpus_mid : corpus_mid + slice_size ]


# Unique characters in the file.  Also said "vocab".
charset = sorted(set(corpus))
print(f"Character set (vocabulary) size: {len(charset)}")


# Layer to convert from tokens to character IDs.
ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary = list(charset), mask_token = None
)
# Inveres of `ids_from_chars'
chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary = ids_from_chars.get_vocabulary(),
        invert = True, mask_token = None
)

def text_from_ids(ids):
    "Turn a character array from model to string."
    return tf.strings.reduce_join(chars_from_ids(ids), axis = -1)


# Convert text vector into a stream of character indices.
all_ids            = ids_from_chars(tf.strings.unicode_split(corpus, 'UTF-8'))
ids_dataset        = tf.data.Dataset.from_tensor_slices(all_ids)
examples_per_epoch = len(corpus) // seq_length + 1
sequences          = ids_dataset.batch(
        seq_length + 1, drop_remainder = True
)


# Turn the dataset into a series of (input, label) for training.
#
# E.g.:
#
# split_input_target(list("Tensorflow"))
# =>
# (['T', 'e', 'n', 's', 'o', 'r', 'f', 'l', 'o'],
#  ['e', 'n', 's', 'o', 'r', 'f', 'l', 'o', 'w'])
#
def split_input_target(sequence):
    input_text  = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

# Create training batches

dataset = (
        sequences.map (split_input_target)
        .shuffle      (training_buffer_size)
        .batch        (training_batch_size, drop_remainder = True)
        .prefetch     (tf.data.experimental.AUTOTUNE)
)


# Build the model

# Here we use Keras to build the model.

model_vocab_size = len(charset)

model = ChomskySaysModel(
        vocabulary_size     = len(ids_from_chars.get_vocabulary()),
        embedding_dimension = model_embed_dim,
        num_rnn_units       = RNN_units
)

# Test the model, first check shape.

for input_ex_batch, target_ex_batch in list(dataset.take(1)):
    ex_batch_predictions = model(input_ex_batch)
    print("Shape (batch size, seq len, vocab size):", ex_batch_predictions.shape)


model.summary()


# Train the model

# Attach optimizer & loss fn

loss               = tf.losses.SparseCategoricalCrossentropy(from_logits = True)
ex_batch_mean_loss = loss(target_ex_batch, ex_batch_predictions)

print("Prediction shape (batch size, seq len, vocab size):\n    ",
        ex_batch_predictions.shape)
print("Mean loss:\n    ",
        ex_batch_mean_loss)

# Configure the training procedure
model.compile(optimizer = 'adam', loss = loss)

# Configure checkpoints

checkpoint_file_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback    = tf.keras.callbacks.ModelCheckpoint(
        filepath          = checkpoint_file_prefix,
        save_weights_only = True
)


# Train

print("Start training...")
history = model.fit(
        dataset, epochs = epochs, callbacks = [checkpoint_callback]
)

# Generate

print("Generate test output...")
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start     = time.time()
states    = None
next_char = tf.constant(['Chomsky says: '])
result    = [next_char]

for n in range(100):
    next_char, states = one_step_model.generate_one_step(
            next_char, states = states
    )
    result.append(next_char)

result = tf.strings.join(result)
end    = time.time()

print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80 + '\n\n')
print("Run time:", end - start)

# Export the model
print('Save model...')
tf.saved_model.save(one_step_model, 'one_step')

