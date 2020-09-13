# compare different dropout schedules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as krs

from Schedule import TrainingSchedule
from Datasets.ImdbDatasets import get_imdb_datasets_fn
from Preprocessings.ImdbPreprocessings import get_imdb_preprocessings_fn
from LearningRates import WarmupLearningRates
from TrainSteps import TrainStepsDefault
from Layers.SingleVectorAddtiveAttention import SingleVectorAddtiveAttention


tf.debugging.set_log_device_placement(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(args):
    schedule = TrainingSchedule()

    # set hyper-parameters
    vocab_size = 10000
    epoches = 20
    max_sentence_length = 512
    training_batch_szie = 200
    eval_batch_szie = 200
    dropout_rate = 0.0
    embedding_dim = 300
    hidden_dim = 300

    # add training datasets and eval datasets
    training_datasets_fn, eval_datasets_fn = get_imdb_datasets_fn(vocab_size)
    schedule.add_training_datasets([training_datasets_fn])
    schedule.add_eval_datasets([eval_datasets_fn])

    # add preprocessings
    preprocessings_fn = get_imdb_preprocessings_fn(max_sentence_length=max_sentence_length)
    schedule.add_preprocessings([preprocessings_fn])

    # add models
    # dropout
    models_lstm_fn_1 = krs.Sequential([
        krs.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                             batch_input_shape=[training_batch_szie, None], input_length=max_sentence_length),
        krs.layers.Dropout(rate=dropout_rate),
        krs.layers.Bidirectional(
            krs.layers.LSTM(units=hidden_dim, stateful=True, return_sequences=True)
        ),
        krs.layers.Dropout(rate=dropout_rate),
        SingleVectorAddtiveAttention(units=hidden_dim),
        krs.layers.Dense(units=hidden_dim, activation="tanh"),
        krs.layers.Dense(1, activation="sigmoid")
    ])

    # drop word
    models_lstm_fn_2 = krs.Sequential([
        krs.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                             batch_input_shape=[training_batch_szie, None], input_length=max_sentence_length),
        krs.layers.Dropout(rate=dropout_rate, noise_shape=[None, None, 1]),
        krs.layers.Bidirectional(
            krs.layers.LSTM(units=hidden_dim, stateful=True, return_sequences=True)
        ),
        krs.layers.Dropout(rate=dropout_rate, noise_shape=[None, None, 1]),
        SingleVectorAddtiveAttention(units=hidden_dim),
        krs.layers.Dense(units=hidden_dim, activation="tanh"),
        krs.layers.Dense(1, activation="sigmoid")
    ])

    # dropout + dropconnection
    models_lstm_fn_3 = krs.Sequential([
        krs.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                             batch_input_shape=[training_batch_szie, None], input_length=max_sentence_length),
        krs.layers.Dropout(rate=dropout_rate, noise_shape=[None, None, 1]),
        krs.layers.Bidirectional(
            krs.layers.LSTM(units=hidden_dim, stateful=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)
        ),
        krs.layers.Dropout(rate=dropout_rate, noise_shape=[None, None, 1]),
        SingleVectorAddtiveAttention(units=hidden_dim),
        krs.layers.Dense(units=hidden_dim, activation="tanh"),
        krs.layers.Dense(1, activation="sigmoid")
    ])

    # drop word + dropconnection
    models_lstm_fn_4 = krs.Sequential([
        krs.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                             batch_input_shape=[training_batch_szie, None], input_length=max_sentence_length),
        krs.layers.Dropout(rate=dropout_rate, noise_shape=[None, None, 1]),
        krs.layers.Bidirectional(
            krs.layers.LSTM(units=hidden_dim, stateful=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)
        ),
        krs.layers.Dropout(rate=dropout_rate, noise_shape=[None, None, 1]),
        SingleVectorAddtiveAttention(units=hidden_dim),
        krs.layers.Dense(units=hidden_dim, activation="tanh"),
        krs.layers.Dense(1, activation="sigmoid")
    ])

    # non-dropout
    models_lstm_fn_5 = krs.Sequential([
        krs.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                             batch_input_shape=[training_batch_szie, None], input_length=max_sentence_length),
        krs.layers.Bidirectional(
            krs.layers.LSTM(units=hidden_dim, stateful=True, return_sequences=True)
        ),
        SingleVectorAddtiveAttention(units=hidden_dim),
        krs.layers.Dense(units=hidden_dim, activation="tanh"),
        krs.layers.Dense(1, activation="sigmoid")
    ])

    schedule.add_models([models_lstm_fn_1, models_lstm_fn_2, models_lstm_fn_3, models_lstm_fn_4, models_lstm_fn_5])

    # add losses
    losses_fn = krs.losses.BinaryCrossentropy(from_logits=False)
    schedule.add_losses([losses_fn])

    # add optimiezers
    optimizers_fn = krs.optimizers.Adam()
    schedule.add_optimizers([optimizers_fn])

    # add learning rates
    learning_rates_fn = WarmupLearningRates(d_model=hidden_dim)
    schedule.add_learning_rates([learning_rates_fn])

    # add train steps
    training_steps_fn = TrainStepsDefault(training_batch_size=training_batch_szie,
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
