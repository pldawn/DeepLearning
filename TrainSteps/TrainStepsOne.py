from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import tensorflow.keras as krs

import TrainSteps.TrainSteps


class DefaultTrainSteps(TrainSteps):
    def __init__(self, training_batch_size, eval_batch_size, epoches, gradient_adjust_fn=lambda x: x):
        super(DefaultTrainSteps, self).__init__(training_batch_size, eval_batch_size, epoches)
        self.gradient_adjust_fn = gradient_adjust_fn

    def call(self, training_datasets_fn, models_fn, losses_fn, optimizers_fn, log_dir, eval_datasets_fn=None,
             learning_rates_fn=None, metrics_fn=None, loss_weights_fn=None, **kwargs):
        # initialize
        training_datasets = training_datasets_fn()
        training_datasets = training_datasets.repeat(self.epoches).shuffle(10000).\
            batch(self.training_batch_size).prefetch(5 * self.training_batch_size)

        if eval_datasets_fn is not None:
            eval_datasets = eval_datasets_fn()
            eval_datasets = eval_datasets.batch(self.eval_batch_size).prefetch(5 * self.eval_batch_size)
        else:
            eval_datasets = None

        loss_weights = None if loss_weights_fn is None else loss_weights_fn()
        metrics = None if metrics_fn is None else metrics_fn()

        if learning_rates_fn is not None:
            optimizers_fn = optimizers_fn.get_config({"learning_rate": learning_rates_fn})

        # initalize callback function
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        checkpoint = os.path.join(log_dir, "model.h5")

        models_fn.compile(optimizer=optimizers_fn, loss=losses_fn, metrics=metrics, loss_weights=loss_weights)

        # training
        steps_per_epoch = self.epoches // self.training_batch_size
        for epoch in range(self.epoches):
            start = time.time()
            step = 0

            for x, y in training_datasets:
                step += 1
                self.one_step(x, y)

                if step >= steps_per_epoch:
                    epoch_spend_time = time.time() - start
                    break

            # evaluate
            if eval_datasets is not None:
                eval_result = models_fn.evaluate(x=eval_datasets, batch_size=self.eval_batch_size)
            else:
                eval_result = []

        result = {"traing_result": training_result, "eval_result": eval_result}

        return result

    @tf.function
    def one_step(self, x, y, models_fn, losses_fn, optimizers_fn, gradient_adjust_fn):
        with tf.GradientTape as tape:
            pred = models_fn(x)
            loss = losses_fn(y, pred)
        gradients = tape.gradient(loss, models_fn.trainable_variables)
        gradients = gradient_adjust_fn(gradients)
        optimizers_fn.apply_gradients(zip(gradients, models_fn.trainable_variables))
