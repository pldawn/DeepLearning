from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class TrainSteps:
    def __init__(self, training_batch_size, eval_batch_size, epoches):
        self.training_batch_size = training_batch_size
        self.eval_batch_size = eval_batch_size
        self.epoches = epoches

    def __call__(self, **kwargs):
        return self.call(**kwargs)

    def call(self, **kwargs):
        raise NotImplementedError

    def one_step(self, **kwargs):
        raise NotImplementedError
