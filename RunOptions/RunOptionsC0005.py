# compare different segmentations
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as krs

from Schedule import TrainingSchedule
from Datasets.TNewsDatasets import get_tnews_datasets_fn
from Preprocessings.TNewsPreprocessings import get_tnews_preprocessings_fn
from Preprocessings import CharTokenizer, JiebaTokenizer, BPETokenizer
from LearingRates import CustomizedLearningRates
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
    dropout_rate = 0.5
    embedding_dim = 300
    hidden_dim = 300
    num_class = 15

    # add training datasets and eval datasets
    training_datasets_fn, eval_datasets_fn, _ = get_tnews_datasets_fn(split=['training', 'validation'])
    schedule.add_training_datasets([training_datasets_fn])
    schedule.add_eval_datasets([eval_datasets_fn])

    # add preprocessings
    char_tokenizer = CharTokenizer()
    jieba_tokenizer = JiebaTokenizer()
    bpe_tokenizer = BPETokenizer()

    preprocessings_fn_1 = get_tnews_preprocessings_fn(max_sentence_length=max_sentence_length,
                                                      tokenizer=char_tokenizer)
    preprocessings_fn_2 = get_tnews_preprocessings_fn(max_sentence_length=max_sentence_length,
                                                      tokenizer=jieba_tokenizer)
    preprocessings_fn_3 = get_tnews_preprocessings_fn(max_sentence_length=max_sentence_length,
                                                      tokenizer=bpe_tokenizer)
    schedule.add_preprocessings([preprocessings_fn_1, preprocessings_fn_2, preprocessings_fn_3])

    # add models
    models_lstm_fn = krs.Sequential([
        krs.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                             batch_input_shape=[training_batch_szie, None], input_length=max_sentence_length),
        krs.layers.Dropout(rate=dropout_rate, noise_shape=[None, None, 1]),
        krs.layers.Bidirectional(
            krs.layers.LSTM(units=hidden_dim, stateful=True, return_sequences=True)
        ),
        krs.layers.Dropout(rate=dropout_rate, noise_shape=[None, None, 1]),
        SingleVectorAddtiveAttention(units=hidden_dim),
        krs.layers.Dense(units=hidden_dim, activation="tanh"),
        krs.layers.Dense(num_class, activation="softmax")
    ])

    schedule.add_models([models_lstm_fn])

    # add losses
    losses_fn = krs.losses.BinaryCrossentropy(from_logits=False)
    schedule.add_losses([losses_fn])

    # add optimiezers
    optimizers_fn = krs.optimizers.Adam()

    schedule.add_optimizers([optimizers_fn])

    # add learning rates
    learning_rates_fn = CustomizedLearningRates(d_model=hidden_dim)
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
