from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as krs

from Schedule import TrainingSchedule
from Datasets.ImdbDatasets import get_imdb_datasets_fn
from Preprocessings.ImdbPreprocessings import get_imdb_preprocessings_fn
from LearingRates import CustomizedLearningRates
from TrainSteps import TrainStepsDefault


tf.debugging.set_log_device_placement(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(args):
    schedule = TrainingSchedule()

    # add training datasets and eval datasets
    training_datasets_fn, eval_datasets_fn, encoder = get_imdb_datasets_fn()
    schedule.add_training_datasets([training_datasets_fn])
    schedule.add_eval_datasets([eval_datasets_fn])

    # add preprocessings
    vocab_size = encoder.vocab_size
    preprocessings_fn = get_imdb_preprocessings_fn(vocab_size=vocab_size, max_sentence_length=512)
    schedule.add_preprocessings([preprocessings_fn])

    # add models
    models_fn = krs.Sequential([
        krs.layers.Embedding(input_dim=vocab_size + 3, output_dim=128, batch_input_shape=[10, None]),
        krs.layers.Dropout(rate=0.5, noise_shape=[None, None, 1]),
        krs.layers.Bidirectional(
            krs.layers.LSTM(units=128, stateful=True, recurrent_initializer="glorot_uniform", return_sequences=True)
        ),
        krs.layers.Dropout(rate=0.5, noise_shape=[None, None, 1]),
        krs.layers.GlobalAveragePooling1D(),
        krs.layers.Dense(units=128, activation="tanh"),
        krs.layers.Dense(1, activation="sigmoid")
    ])
    schedule.add_models([models_fn])

    # add losses
    losses_fn = krs.losses.SparseCategoricalCrossentropy(from_logits=False)
    schedule.add_losses([losses_fn])

    # add optimiezers
    optimizers_fn = krs.optimizers.Adam()
    schedule.add_optimizers([optimizers_fn])

    # add learning rates
    learning_rates_fn = CustomizedLearningRates(d_model=128)
    schedule.add_learning_rates([learning_rates_fn])

    # add train steps
    training_steps_fn = TrainStepsDefault(training_batch_size=10, eval_batch_size=10, epoches=2)
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
