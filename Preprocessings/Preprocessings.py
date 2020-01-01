from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras as krs
import tensorflow_datasets as tfds


def add_eos_to_texts(inputs, start="<START>", end="<END>"):
    output = [[start] + list(inp) + [end] for inp in inputs]

    return output


def add_eos_to_ids(inputs, start=1, end=2):
    output = [[start] + list(inp) + [end] for inp in inputs]

    return output


def ordinal_encode_1d(labels, label_table=None):
    if label_table is None:
        unique_labels = set(labels)
        label_table = {}

        for ind, label in enumerate(unique_labels):
            label_table[label] = ind

    ordinal_labels = [label_table[label] for label in labels]

    return ordinal_labels, label_table


def ordinal_encode_2d(labels, all_label_tables=None):
    kinds_num = len(labels[0])
    all_ordinal_labels = []

    if all_label_tables is None:
        all_label_tables = [None] * kinds_num

    all_kinds_labels = list(zip(*labels))

    for ind in range(len(all_kinds_labels)):
        one_kind_labels = all_kinds_labels[ind]
        label_table = all_label_tables[ind]
        one_kind_ordinal_labels, one_kind_label_table = ordinal_encode_1d(one_kind_labels, label_table=label_table)
        all_ordinal_labels.append(one_kind_ordinal_labels)

        if all_label_tables is None:
            all_label_tables[ind] = one_kind_label_table

    return all_ordinal_labels, all_label_tables
