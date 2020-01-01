from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Preprocessings import Tokenizer
import jieba


class JiebaTokenizer(Tokenizer):
    def segment(self, inputs):
        output = []

        for sentence in inputs:
            tokens = jieba.lcut(sentence)
            output.append(tokens)

        return output
