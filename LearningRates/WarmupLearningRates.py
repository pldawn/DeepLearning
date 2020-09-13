from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as krs
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


# learning_rate = (d_model * -0.5) * min(step_num ** -0.5, step_num * warm_up_steps ** -1.5))
class WarmupLearningRates(krs.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warm_up_steps=4000):
        super(WarmupLearningRates, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warm_up_steps = tf.cast(warm_up_steps, tf.float32)

    def __call__(self, step):
        args1 = tf.math.rsqrt(step)
        args2 = step * (self.warm_up_steps ** -1.5)
        args3 = tf.math.rsqrt(self.d_model)

        lr = args3 * tf.math.minimum(args1, args2)

        return lr

    def get_config(self):
        return self.__dict__
