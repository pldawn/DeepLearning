# compare different initializers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as krs

from Schedule import TrainingSchedule
from Datasets.ImdbDatasets import get_imdb_datasets_fn
from Preprocessings.ImdbPreprocessings import get_imdb_preprocessings_fn
from LearningRates import WarmupLearningRates
from Layers import MultiplicativeDense
from Losses import FocusLossForSingleTask
from TrainSteps import TrainStepsForMultiTask


tf.debugging.set_log_device_placement(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(args):
    schedule = TrainingSchedule()

    # set hyper-parameters
    feature_size = 10000
    epoches = 20
    max_sentence_length = 512
    training_batch_szie = 200
    eval_batch_szie = 200
    dropout_rate = 0.0
    embedding_dim = 300
    hidden_dim = 300
    layer_nums = 3
    task_nums = 6

    # add training datasets and eval datasets
    training_datasets_fn, eval_datasets_fn = get_imdb_datasets_fn(feature_size)
    schedule.add_training_datasets([training_datasets_fn])
    schedule.add_eval_datasets([eval_datasets_fn])

    # add preprocessings
    preprocessings_fn = get_imdb_preprocessings_fn(max_sentence_length=max_sentence_length)
    schedule.add_preprocessings([preprocessings_fn])

    # add models
    inputs = krs.layers.Input(shape=(feature_size,))
    features = inputs

    for _ in range(layer_nums):
        addtive_features = krs.layers.Dense(units=hidden_dim//2, activation=krs.activations.tanh, use_bias=True)(features)
        multiplicative_features = MultiplicativeDense(units=hidden_dim//2, activation=krs.activations.tanh)(features)
        features = tf.concat([addtive_features, multiplicative_features], axis=-1)

    outputs = []

    for _ in range(task_nums):
        hidden_1 = krs.layers.Dense(units=hidden_dim, activation=krs.activations.tanh, use_bias=True)(features)
        hidden_2 = krs.layers.Dense(units=hidden_dim, activation=krs.activations.tanh, use_bias=True)(hidden_1)
        logits = krs.layers.Dense(units=2)(hidden_2)
        outputs.append(logits)

    models_lstm_fn = krs.models.Model(inputs=inputs, outputs=outputs)
    schedule.add_models([models_lstm_fn])

    # add losses
    losses_fn = FocusLossForSingleTask(from_logits=True)
    schedule.add_losses([losses_fn])

    # add optimiezers
    optimizers_fn = krs.optimizers.Adam()
    schedule.add_optimizers([optimizers_fn])

    # add learning rates
    learning_rates_fn = WarmupLearningRates(d_model=hidden_dim)
    schedule.add_learning_rates([learning_rates_fn])

    # add train steps
    training_steps_fn = TrainStepsForMultiTask(training_batch_size=training_batch_szie,
                                               eval_batch_size=eval_batch_szie, epoches=epoches)
    schedule.add_training_steps([training_steps_fn])

    # add eval steps

    # add metrics
    def metrics_fn():
        metrcis = [
            krs.metrics.Precision(),
            krs.metrics.Recall(),
            krs.metrics.AUC()
        ]

        return metrcis

    schedule.add_metrics([metrics_fn])

    results = schedule.run()


if __name__ == "__main__":
    main([])
