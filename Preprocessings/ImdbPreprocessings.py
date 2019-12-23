from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as krs


def get_imdb_preprocessings_fn(max_sentence_length=512):

    def imdb_preprocessings_fn(datasets_fn):
        datasets, labels = datasets_fn()

        # pad sentences
        datasets = krs.preprocessing.sequence.pad_sequences(
            datasets, value=3, padding='post', maxlen=max_sentence_length)

        # define datasets function
        def datasets_fn():
            return datasets, labels

        return datasets_fn

    return imdb_preprocessings_fn
