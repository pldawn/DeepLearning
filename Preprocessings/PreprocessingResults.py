from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class PreprocessingResults:
    def __init__(self):
        self.datasets_fn = None
        self.vocabulary = None
        self.label_table = None

    def get(self, discard=None):
        assert discard is None or type(discard) == list

        result = self.__dict__.copy()

        if discard is not None:
            for key in discard:
                del result[key]

        return result

    def get_datasets_fn(self):
        return self.datasets_fn

    def get_vocabulary(self):
        return self.vocabulary

    def get_label_table(self):
        return self.label_table

    def set_datasets_fn(self, datasets_fn):
        self.datasets_fn = datasets_fn

    def set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary

    def set_label_table(self, label_table):
        self.label_table = label_table
