from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as krs


class FocusLossForSingleTask:
    def __init__(self, alpha=4.0, gama=0.5, sparse_labels=True, **kwargs):
        self.alpha = alpha
        self.gama = gama
        self.sparse_labels = sparse_labels
        self.kwargs = kwargs

        if self.sparse_labels:
            self.core_loss_fn = krs.losses.SparseCategoricalCrossentropy(**kwargs)
        else:
            self.core_loss_fn = krs.losses.CategoricalCrossentropy(**kwargs)

    def __call__(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.convert_to_tensor(y_pred)

        if self.kwargs.get("from_logits", False):
            y_pred_prob = krs.activations.softmax(y_pred)
            print(y_pred_prob)
        else:
            y_pred_prob = y_pred

        if self.sparse_labels:
            if y_true.ndim > 1:
                y_true = tf.squeeze(y_true)

            sample_weight = [1 - y_pred_prob[i][y_true[i]] for i in range(len(y_true))]

        else:
            sample_weight = [1 - tf.reduce_sum(tf.cast(y_true[i], y_pred_prob.dtype) * y_pred_prob[i])
                             for i in range(len(y_true))]

        sample_weight = tf.convert_to_tensor(sample_weight)
        sample_weight = self.alpha * sample_weight ** self.gama

        loss = self.core_loss_fn(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

        return loss


class GradNorm:
    """
    reference: https://arxiv.org/abs/1711.02257
    blog: https://blog.csdn.net/Leon_winter/article/details/105014677
    """

    def __init__(self, alpha=0.12, loss_zero=None):
        self.alpha = alpha
        self.loss_zero = loss_zero
        self.T = 0

        if self.loss_zero is not None:
            self.T = len(loss_zero)

    def set_loss_zero(self, loss_zero):
        self.loss_zero = loss_zero
        self.T = len(loss_zero)

    def normalize(self, x, w, grad, loss, optimizer=tf.optimizers.SGD()):
        if self.loss_zero is None or self.T == 0:
            raise AttributeError("loss_zero is None or grad nums is zero")

        # step 1: 计算所有任务对x的总梯度
        total_grad = tf.reduce_sum([w[i] * grad[i] for i in range(self.T)], axis=1)

        # step 2: 计算各任务梯度的r、norm和average norm
        with tf.GradientTape() as tape_w:
            relative_loss = [loss[i] / self.loss_zero[i] for i in range(self.T)]
            relative_loss_mean = tf.reduce_mean(relative_loss)
            r = [relative_loss[i] / relative_loss_mean for i in range(self.T)]

            grad_norm = [tf.norm(w[i] * grad[i]) for i in range(self.T)]
            grad_norm_mean = tf.reduce_mean(grad_norm)

        # step 3: 计算Grad Loss
            grad_norm_abs = [tf.abs(grad_norm[i] - grad_norm_mean * r[i] ** self.alpha) for i in range(self.T)]
            grad_loss = tf.reduce_sum(grad_norm_abs)

        # step 4: 计算Grad Loss对w的梯度
        w_grad = tape_w.gradient(grad_loss, w)

        # step 5: 利用总梯度更新x
        optimizer.apply_gradients(zip(total_grad, x))

        # step 6: 利用Grad Loss更新w
        optimizer.apply_gradients(zip(w_grad, w))

        # step 7: 对w进行normalization，使其求和等于任务数量
        scaling = self.T / tf.reduce_sum(w)
        w = [w[i] * scaling for i in range(self.T)]

        return w


if __name__ == "__main__":
    with tf.device("/cpu:0"):
        grad_norm_fn = GradNorm(loss_zero=[tf.convert_to_tensor(1.0), tf.convert_to_tensor(1.0)])
        x = tf.Variable(1.0)
        y = tf.Variable(2.0)

        with tf.GradientTape(persistent=True) as tape:
            z1 = 2 * x + y
            z2 = x + y ** 2

        xs = [x, y]
        loss = [z1, z2]
        grad = []

        for l in loss:
            grad.append(tape.gradient(l, xs))

        ws = [tf.Variable(1.0, trainable=True), tf.Variable(1.0, trainable=True)]

        print(xs)
        print(ws)

        ws = grad_norm_fn.normalize(x=xs, w=ws, grad=grad, loss=loss)

        print(xs)
        print(ws)