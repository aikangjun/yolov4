# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/6 14:06
@Project : YOLOP
@File : customlosses.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import numpy as np
from custom import *
import tensorflow as tf


class EagerMaskedReducalNLL(losses.Loss):
    """
    calculate yolo classification error based on NLL
    by using object mask
    """

    def __init__(self,
                 mask: np.ndarray,
                 target_num: int,
                 **kwargs):
        super(EagerMaskedReducalNLL, self).__init__(**kwargs)
        assert self.reduction in [tf.losses.Reduction.AUTO, tf.losses.Reduction.SUM]
        self.mask = tf.cast(mask, dtype=tf.bool)
        self.target_num = target_num

    def call(self, y_true, y_pred):
        """
        in eager execution,
        do not use tf's logical operators、iterators
        """
        masked_true_prob = tf.boolean_mask(y_true, self.mask)
        masked_pred_prob = tf.boolean_mask(y_pred, self.mask)
        nll_loss = tf.abs(masked_pred_prob) * masked_true_prob

        target_size = tf.shape(nll_loss)[0]
        if tf.equal(target_size, 0) or tf.equal(target_size, 1):
            return tf.reduce_sum(nll_loss)

        # obtain negative mask by determining the num of true boxes of same class
        neg_mask = [tf.greater(tf.reduce_sum(masked_prob), tf.cast(target_size // 2, dtype=tf.float32))
                    for masked_prob in tf.split(masked_true_prob,
                                                num_or_size_splits=self.target_num, axis=-1)]

        # sieve out negative error
        neg_loss = tf.boolean_mask(nll_loss, neg_mask, axis=1)
        neg_loss = tf.reshape(neg_loss, shape=[-1])
        _, indices = tf.nn.top_k(neg_loss,
                                 k=tf.cast(target_size // 2, dtype=tf.int32)
                                 if tf.reduce_any(neg_mask) else 0)
        neg_loss = tf.gather(neg_loss, indices)

        pos_mask = tf.logical_not(neg_mask)
        pos_loss = tf.boolean_mask(nll_loss, pos_mask, axis=1)
        pos_loss = tf.reduce_sum(pos_loss, axis=-1)

        total_loss = tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)
        if self.reduction == tf.losses.Reduction.AUTO:
            neg_size = tf.shape(neg_loss)[0]
            pos_size = tf.shape(pos_loss)[0]
            total_loss /= tf.cast(neg_size + pos_size, dtype=tf.float32)

        return total_loss
