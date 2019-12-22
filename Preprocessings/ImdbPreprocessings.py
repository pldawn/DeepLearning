from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds
from Preprocessings.Preprocessings import *


def get_imdb_preprocessings_fn(vocab_size, datasets_fn_config="subwords32k", max_sentence_length=512,
                               add_eos_fn=add_eos, num_parallel_calls=5):

    def imdb_preprocessings_fn(datasets_fn, training=True):
        datasets = datasets_fn()

        if datasets_fn_config == "plain_text":
            pass

        # add <SOS> and <EOS> to sentences
        if add_eos_fn is not None:
            datasets = datasets.map(lambda x: add_eos_fn(x, vocab_size=vocab_size, with_label=True),
                                    num_parallel_calls=num_parallel_calls)
            padding_value = vocab_size + 2
        else:
            padding_value = vocab_size

        # pad sentences
        datasets = datasets.map(lambda x: pad_sequence(x, max_sentence_length=max_sentence_length,
                                                       padding_value=padding_value, with_label=True),
                                num_parallel_calls=num_parallel_calls)

        # convert {'text': x, 'label': y} to (x, y)
        datasets = datasets.map(lambda data: (data['text'], data['label']))

        # define datasets function
        def datasets_fn():
            return datasets

        return datasets_fn

    return imdb_preprocessings_fn
