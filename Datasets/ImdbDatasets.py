from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.datasets.imdb as imdb


def get_imdb_datasets_fn(vocab_size, **kwargs):
    # get raw datasets
    (training_data, training_label), (eval_data, eval_label) = imdb.load_data(num_words=vocab_size)

    # define datasets function
    def training_datasets_fn():
        return training_data, training_label

    def test_datasets_fn():
        return eval_data, eval_label

    return training_datasets_fn, test_datasets_fn
