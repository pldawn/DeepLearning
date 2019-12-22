from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as krs
import tensorflow_datasets as tfds


def add_eos(example, vocab_size, with_label):
    if with_label:
        new_exapmle = {}

        for k in example:
            if k == 'label':
                new_exapmle[k] = example[k]
            else:
                new_exapmle[k] = tf.concat([[vocab_size], example[k], [vocab_size + 1]], axis=0)

    else:
        new_exapmle = tf.concat([[vocab_size], example, [vocab_size + 1]], axis=0)

    return new_exapmle


def pad_sequence(example, max_sentence_length, padding_value, with_label):
    if with_label:
        new_example = {}

        for k in example:
            if k == 'label':
                new_example[k] = example[k]
            else:
                pad = [padding_value] * max_sentence_length
                new_example[k] = tf.concat([example[k], pad], axis=0)
                new_example[k] = new_example[k][:max_sentence_length]

    else:
        pad = [padding_value] * max_sentence_length
        new_example = tf.concat([example, pad], axis=0)
        new_example = new_example[:max_sentence_length]

    return new_example
