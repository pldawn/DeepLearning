from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.keras as krs
from Preprocessings.Preprocessings import add_eos_to_ids


class Tokenizer:
    def __init__(self, vocabulary=None, **kwargs):
        self.vocabulary = vocabulary
        self.tokenzier = None
        self.counter = None

    def segment(self, inputs):
        raise NotImplementedError

    def tokenize(self, inputs, padding=True, pad_id=0, start_id=1, end_id=2, unk_id=3, first_token_id=4,
                 max_sentence_length=512, eos=True, min_df=1, lowercase=False, **kwargs):
        # convert to lowercase
        if lowercase:
            inputs = [sent.lower() for sent in inputs]

        # segment texts to tokens
        tokenized_inputs = self.segment(inputs)

        # build vocabulary
        if self.vocabulary is None:
            self.vocabulary = self.bulid_vocabulary(tokenized_inputs=tokenized_inputs, first_token_id=first_token_id,
                                                    pad_id=pad_id, start_id=start_id, end_id=end_id, unk_id=unk_id,
                                                    min_df=min_df)

        # map texts to ids
        ids = self.texts_to_ids(tokenized_inputs=tokenized_inputs, vocabulary=self.vocabulary)

        # add eos
        if eos:
            ids = add_eos_to_ids(ids, start=start_id, end=end_id)

        # padding
        if padding:
            ids = krs.preprocessing.sequence.pad_sequences(sequences=ids, maxlen=max_sentence_length,
                                                           padding="post", value=pad_id)

        return ids

    def bulid_vocabulary(self, tokenized_inputs, first_token_id, pad_id, unk_id, start_id, end_id, min_df=1):
        counter = {}
        vocabulary = {}
        token_id = first_token_id

        for tokenized_input in tokenized_inputs:
            for token in tokenized_input:
                counter[token] = counter.setdefault(token, 0) + 1

        for token, freq in counter.items():
            if freq >= min_df:
                vocabulary[token] = token_id
                token_id += 1

        vocabulary['<PAD>'] = pad_id
        vocabulary['<UNK>'] = unk_id
        vocabulary['<START>'] = start_id
        vocabulary['<END>'] = end_id

        return vocabulary

    def texts_to_ids(self, tokenized_inputs, vocabulary, unk_id=0):
        ids = [[vocabulary.get(token, unk_id) for token in tokenized_input] for tokenized_input in tokenized_inputs]

        return ids

    def get_vocabulary(self):
        return self.vocabulary
