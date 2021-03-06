# compare different learning rate schedules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import tensorflow.keras as krs

from Schedule import TrainingSchedule
from Pipelines import TrainingPipelines
from Datasets.TNewsDatasets import get_tnews_datasets_fn
from Preprocessings.TNewsPreprocessings import get_tnews_preprocessings_fn
from Preprocessings import BPETokenizer
from LearningRates import WarmupLearningRates
from TrainSteps import TrainStepsDefault, TrainStepsReduceLROnPlateau
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
    bpe_tokenizer = BPETokenizer()
    preprocessings_fn = get_tnews_preprocessings_fn(max_sentence_length=max_sentence_length,
                                                    tokenizer=bpe_tokenizer)
    schedule.add_preprocessings([preprocessings_fn])

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
    losses_fn = krs.losses.SparseCategoricalCrossentropy(from_logits=False)
    schedule.add_losses([losses_fn])

    # add optimiezers
    optimizers_fn_1 = krs.optimizers.SGD()
    optimizers_fn_2 = krs.optimizers.Adam()
    optimizers_fn_3 = krs.optimizers.Adam(decay=1e-5)

    schedule.add_optimizers([optimizers_fn_1, optimizers_fn_2, optimizers_fn_3])

    # add learning rates
    # constant
    learning_rates_fn_1 = 1 / math.sqrt(hidden_dim)

    # warmup
    learning_rates_fn_2 = WarmupLearningRates(d_model=hidden_dim)

    schedule.add_learning_rates([learning_rates_fn_1, learning_rates_fn_2])

    # add train steps
    training_steps_fn_1 = TrainStepsDefault(training_batch_size=training_batch_szie,
                                            eval_batch_size=eval_batch_szie, epoches=epoches)
    training_steps_fn_2 = TrainStepsReduceLROnPlateau(training_batch_size=training_batch_szie,
                                                      eval_batch_size=eval_batch_szie, epoches=epoches)

    schedule.add_training_steps([training_steps_fn_1, training_steps_fn_2])

    # add eval steps

    # add metrics
    def metrics_fn():
        metrcis = [
            krs.metrics.sparse_categorical_accuracy
        ]

        return metrcis

    schedule.add_metrics([metrics_fn])

    # add pipelines
    tp1 = TrainingPipelines()
    tp2 = TrainingPipelines()
    tp3 = TrainingPipelines()
    tp4 = TrainingPipelines()

    tp1.add_process_method_num('training_datasets', 1)
    tp1.add_process_method_num('eval_datasets', 1)
    tp1.add_process_method_num('preprocessings', 1)
    tp1.add_process_method_num('models', 1)
    tp1.add_process_method_num('losses', 1)
    tp1.add_process_method_num('optimizers', 3)
    tp1.add_process_method_num('learning_rates', 2)
    tp1.add_process_method_num('training_steps', 2)
    tp1.add_process_method_num('metrics', 1)

    tp2.add_process_method_num('training_datasets', 1)
    tp2.add_process_method_num('eval_datasets', 1)
    tp2.add_process_method_num('preprocessings', 1)
    tp2.add_process_method_num('models', 1)
    tp2.add_process_method_num('losses', 1)
    tp2.add_process_method_num('optimizers', 3)
    tp2.add_process_method_num('learning_rates', 2)
    tp2.add_process_method_num('training_steps', 2)
    tp2.add_process_method_num('metrics', 1)

    tp3.add_process_method_num('training_datasets', 1)
    tp3.add_process_method_num('eval_datasets', 1)
    tp3.add_process_method_num('preprocessings', 1)
    tp3.add_process_method_num('models', 1)
    tp3.add_process_method_num('losses', 1)
    tp3.add_process_method_num('optimizers', 3)
    tp3.add_process_method_num('learning_rates', 2)
    tp3.add_process_method_num('training_steps', 2)
    tp3.add_process_method_num('metrics', 1)

    tp4.add_process_method_num('training_datasets', 1)
    tp4.add_process_method_num('eval_datasets', 1)
    tp4.add_process_method_num('preprocessings', 1)
    tp4.add_process_method_num('models', 1)
    tp4.add_process_method_num('losses', 1)
    tp4.add_process_method_num('optimizers', 3)
    tp4.add_process_method_num('learning_rates', 2)
    tp4.add_process_method_num('training_steps', 2)
    tp4.add_process_method_num('metrics', 1)

    tp1.add_process('training_datasets', 'ALL')
    tp1.add_process('eval_datasets', 'ALL')
    tp1.add_process('preprocessings', 'ALL')
    tp1.add_process('models', 'ALL')
    tp1.add_process('losses', 'ALL')
    tp1.add_process('optimizers', 0)
    tp1.add_process('learning_rates', 'ALL')
    tp1.add_process('training_steps', 0)
    tp1.add_process('metrics', 'ALL')

    tp2.add_process('training_datasets', 'ALL')
    tp2.add_process('eval_datasets', 'ALL')
    tp2.add_process('preprocessings', 'ALL')
    tp2.add_process('models', 'ALL')
    tp2.add_process('losses', 'ALL')
    tp2.add_process('optimizers', 0)
    tp2.add_process('learning_rates', 0)
    tp2.add_process('training_steps', 1)
    tp2.add_process('metrics', 'ALL')

    tp3.add_process('training_datasets', 'ALL')
    tp3.add_process('eval_datasets', 'ALL')
    tp3.add_process('preprocessings', 'ALL')
    tp3.add_process('models', 'ALL')
    tp3.add_process('losses', 'ALL')
    tp3.add_process('optimizers', 1)
    tp3.add_process('learning_rates', 'ALL')
    tp3.add_process('training_steps', 0)
    tp3.add_process('metrics', 'ALL')

    tp4.add_process('training_datasets', 'ALL')
    tp4.add_process('eval_datasets', 'ALL')
    tp4.add_process('preprocessings', 'ALL')
    tp4.add_process('models', 'ALL')
    tp4.add_process('losses', 'ALL')
    tp4.add_process('optimizers', 2)
    tp4.add_process('learning_rates', 0)
    tp4.add_process('training_steps', 0)
    tp4.add_process('metrics', 'ALL')

    tp4.add_sub_pipelines(tp1)
    tp4.add_sub_pipelines(tp2)
    tp4.add_sub_pipelines(tp3)

    schedule.add_pipelines(tp4)

    results = schedule.run()

krs.layers.LSTM
if __name__ == "__main__":
    main([])
