from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as krs
from Preprocessings.Preprocessings import ordinal_encode_1d
from Preprocessings import PreprocessingResults


def get_tnews_preprocessings_fn(max_sentence_length=512, tokenizer=None):

    def tnews_preprocessings_fn(datasets_fn, label_table=None, **kwargs):
        result = PreprocessingResults()
        datasets, labels = datasets_fn()

        if tokenizer is not None:
            datasets = tokenizer.tokenize(datasets, max_sentence_length=max_sentence_length)

        labels, label_table = ordinal_encode_1d(labels=labels, label_table=label_table)

        def new_datasets_fn():
            return datasets, labels

        result.set_datasets_fn(new_datasets_fn)
        result.set_vocabulary(tokenizer.get_vocabulary())
        result.set_label_table(label_table)

        return result

    return tnews_preprocessings_fn
