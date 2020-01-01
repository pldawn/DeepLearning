from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as krs
from Preprocessings import PreprocessingResults


def get_imdb_preprocessings_fn(max_sentence_length=512, **kwargs):

    def imdb_preprocessings_fn(datasets_fn, **kwargs):
        result = PreprocessingResults()
        datasets, labels = datasets_fn()

        # pad sentences
        datasets = krs.preprocessing.sequence.pad_sequences(
            datasets, value=0, padding='post', maxlen=max_sentence_length)

        # define datasets function
        def datasets_fn():
            return datasets, labels

        result.set_datasets_fn(datasets_fn)

        return result

    return imdb_preprocessings_fn
