from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds

from Datasets.Datasets import load_tensorflow_datasets


def get_imdb_datasets_fn(config="subwords32k", split=None, **kwargs):
    assert config in ["plain_text", "bytes", "subwords8k", "subwords32k"], \
        "valid values: plain_text, bytes, subwords8k, subwords32k"

    # get raw datasets
    builder_kwargs = {"config": config}
    datasets, info = load_tensorflow_datasets("imdb_reviews", split=split, builder_kwargs=builder_kwargs)

    training_datasets = datasets['train']
    test_datasets = datasets['test']
    encoder = info.features['text'].encoder

    # define datasets function
    def training_datasets_fn():
        return training_datasets

    def test_datasets_fn():
        return test_datasets

    return training_datasets_fn, test_datasets_fn, encoder
