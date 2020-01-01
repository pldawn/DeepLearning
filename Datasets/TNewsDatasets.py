from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json


tnews_training_file = r"D:\PythonProject\DeepLearning\resources\tnews_public\train.json"
tnews_validation_file = r"D:\PythonProject\DeepLearning\resources\tnews_public\dev.json"
tnews_test_file = r"D:\PythonProject\DeepLearning\resources\tnews_public\test.json"


def get_tnews_datasets_fn(split=None, **kwargs):
    if isinstance(split, list):
        for key in split:
            assert key in ['training', 'validation', 'test']
    elif split is None:
        split = ['training', 'validation', 'test']
    else:
        assert isinstance(split, list) or split is None

    training_data, training_label = [], []
    validation_data, validation_label = [], []
    test_data, test_label = [], []

    if 'training' in split:
        training_data, training_label = load_tnews_file(tnews_training_file)

    if 'validation' in split:
        validation_data, validation_label = load_tnews_file(tnews_validation_file)

    if 'test' in split:
        test_data, test_label = load_tnews_file(tnews_test_file)

    def training_datasets_fn():
        return training_data, training_label

    def validation_datasets_fn():
        return validation_data, validation_label

    def test_datasets_fn():
        return test_data, test_label

    return training_datasets_fn, validation_datasets_fn, test_datasets_fn


def load_tnews_file(filename, encoding='utf-8'):
    data, label = [], []

    with open(filename, encoding=encoding) as f:
        for line in f:
            line = json.loads(line.strip())
            data.append(line.get('sentence', ''))
            label.append(line.get('label', "-1"))

    return data, label
