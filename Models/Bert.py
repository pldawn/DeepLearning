from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow as tf
from bert4keras.backend import keras, set_gelu
from bert4keras.bert import build_bert_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr


set_gelu("tanh")

tf.debugging.set_log_device_placement(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

root = r"D:\PythonProject\DeepLearning\resources\albert_small_zh_google"
config_path = os.path.join(root, "albert_config_small_google.json")
checkpoint_path = os.path.join(root, "albert_model.ckpt")

bert = build_bert_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='albert',
    with_pool=True,
    return_keras_model=False,
)

output = keras.layers.Dropout(rate=0.1)(bert.model.output)
output = keras.layers.Dense(units=1,
                            activation='sigmoid',
                            kernel_initializer=bert.initializer)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()
AdamLR = extend_with_piecewise_linear_lr(Adam)

model.compile(
    loss='binary_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=AdamLR(learning_rate=1e-4,
                     lr_schedule={1000: 1, 2000: 0.1}),
    metrics=['accuracy'],
)


