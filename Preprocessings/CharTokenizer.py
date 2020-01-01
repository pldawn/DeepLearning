from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Preprocessings import Tokenizer
import re


class CharTokenizer(Tokenizer):
    def segment(self, inputs):
        output =[]
        pattern = re.compile('[^0-9a-zA-Z\\s]|[0-9a-zA-Z]+')

        for sentence in inputs:
            tokens = re.findall(pattern=pattern, string=sentence)
            output.append(tokens)

        return output
