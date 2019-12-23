from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import tensorflow.keras as krs

from TrainSteps import TrainSteps


class TrainStepsDefault(TrainSteps):
    def __init__(self, training_batch_size, eval_batch_size, epoches, gradient_adjust_fn=lambda x: x):
        super(TrainStepsDefault, self).__init__(training_batch_size, eval_batch_size, epoches)
        self.gradient_adjust_fn = gradient_adjust_fn

    def call(self, training_datasets_fn, models_fn, losses_fn, optimizers_fn, log_dir, eval_datasets_fn=None,
             learning_rates_fn=None, metrics_fn=None, loss_weights_fn=None, **kwargs):
        # initialize
        training_datasets, training_labels = training_datasets_fn()

        if eval_datasets_fn is not None:
            eval_datasets, eval_labels = eval_datasets_fn()
        else:
            eval_datasets, eval_labels = None, None

        loss_weights = None if loss_weights_fn is None else loss_weights_fn()
        metrics = None if metrics_fn is None else metrics_fn()

        if learning_rates_fn is not None:
            optimizers_fn = optimizers_fn.from_config(config={"learning_rate": learning_rates_fn})

        # initalize callback function
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        checkpoint = os.path.join(log_dir, "model.h5")

        callbacks = [
            krs.callbacks.TensorBoard(log_dir),
            krs.callbacks.ModelCheckpoint(checkpoint, save_best_only=False),
            krs.callbacks.EarlyStopping(patience=10, min_delta=1e-3)
        ]

        models_fn.compile(optimizer=optimizers_fn, loss=losses_fn, metrics=metrics, loss_weights=loss_weights)

        # training
        training_result = models_fn.fit(x=training_datasets, y=training_labels, epochs=self.epoches,
                                        batch_size=self.training_batch_size, callbacks=callbacks,
                                        validation_split=0.1)

        # evaluate
        if eval_datasets is not None and eval_labels is not None:
            eval_result = models_fn.evaluate(x=eval_datasets, y=eval_labels, batch=self.eval_batch_size)
        else:
            eval_result = {}

        result = {"traing_result": training_result, "eval_result": eval_result}

        return result

    def one_step(self, **kwargs):
        pass
