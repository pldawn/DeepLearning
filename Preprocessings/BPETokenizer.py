from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Preprocessings import Tokenizer
from bpemb import BPEmb
from bpe import Encoder


class BPETokenizer(Tokenizer):
    def __init__(self, pre_train=True, vs=100000, vocab_size=10000, pct_bpe=0.8, vocabulary=None, **kwargs):
        super(BPETokenizer, self).__init__(vocabulary, **kwargs)
        self.pre_train = pre_train
        self.vs = vs
        self.vocab_size = vocab_size
        self.pct_bpe = pct_bpe

    def segment(self, inputs):
        output = []

        # use pre-train bpe vocabulary
        if self.pre_train:
            bpemb = BPEmb(lang="zh", vs=self.vs)

            for sentence in inputs:
                tokens = bpemb.encode(sentence)
                output.append(tokens)

        # train a bpe vocabulary corresponding to inputs
        else:
            encoder = Encoder(vocab_size=self.vocab_size, pct_bpe=self.pct_bpe, ngram_max=10)
            encoder.fit(inputs)

            for sentence in inputs:
                tokens = encoder.tokenize(sentence)
                output.append(tokens)

        return output
