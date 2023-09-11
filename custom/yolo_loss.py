# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/6 14:32
@Project : YOLOP
@File : yolo_loss.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import math
from custom import *
from custom.customlosses import EagerMaskedReducalNLL
import tensorflow as tf

binary_crossentropy = losses.BinaryCrossentropy(from_logits=True,
                                                reduction=tf.losses.Reduction.NONE)


def smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(tf.shape(y_true)[-1], dtype=tf.float32)
    label_smoothing = tf.cast(label_smoothing, dtype=tf.float32)
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


def yolo_head(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    # ---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    # ---------------------------------------------------#
    anchors_tensor = tf.reshape(tf.cast(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])

    # ---------------------------------------------------#
    #   获得x，y的网格
    #   (13, 13, 1, 2)
    # ---------------------------------------------------#
    grid_shape = tf.shape(feats)[1:3]
    grids = tf.meshgrid(tf.range(grid_shape[1]), tf.range(grid_shape[0]))
    grid_xy = tf.stack(grids, axis=-1)
    grid = grid_xy[..., tf.newaxis, :]
    grid = tf.cast(grid, dtype=feats.dtype)

    # ---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,k+5)
    #   k代表的是种类的置信度
    # ---------------------------------------------------#
    feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # ---------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy对应框的中心点
    #   box_wh对应框的宽和高
    # ---------------------------------------------------#
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[..., ::-1], dtype=feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[..., ::-1], dtype=feats.dtype)

    # 下面的操作非常重要，能够帮助快速收敛，并且能够防止梯度爆炸，梯度消失导致的nan
    # !!!
    box_confidence = feats[..., 4:5]
    box_class_probs = tf.nn.log_softmax(feats[..., 5:], axis=-1)

    return box_xy, box_wh, box_confidence, box_class_probs


def box_iou(b1, b2):
    # 13,13,3,1,4
    # 计算左上角的坐标和右下角的坐标
    b1 = tf.expand_dims(b1, axis=-2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 1,n,4
    # 计算左上角和右下角的坐标
    b2 = tf.expand_dims(b2, axis=0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 通过拓展域, 计算重叠面积, pred:(13, 13, 3) → 1, 1 → n
    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_ciou(b1, b2):
    """
    :param b1: shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    :param b2: shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    :return: ciou, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # -----------------------------------------------------------#
    #   求出预测框左上角右下角
    #   b1_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b1_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    # -----------------------------------------------------------#
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # -----------------------------------------------------------#
    #   求出真实框左上角右下角
    #   b2_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b2_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    # -----------------------------------------------------------#
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # -----------------------------------------------------------#
    #   求真实框和预测框所有的iou
    #   iou         (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / tf.maximum(union_area, 1e-7)

    # -----------------------------------------------------------#
    #   计算中心的差距
    #   center_distance (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    center_distance = tf.reduce_sum(tf.square(b1_xy - b2_xy), axis=-1)
    enclose_mins = tf.minimum(b1_mins, b2_mins)
    enclose_maxes = tf.maximum(b1_maxes, b2_maxes)
    enclose_wh = tf.maximum(enclose_maxes - enclose_mins, 0.0)
    # -----------------------------------------------------------#
    #   计算对角线距离
    #   enclose_diagonal (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)
    ciou = iou - 1.0 * (center_distance) / tf.maximum(enclose_diagonal, 1e-7)  # 防止无检测体时, enclose距离为0的情况

    v = 4 * tf.square(tf.math.atan2(b1_wh[..., 0], tf.maximum(b1_wh[..., 1], 1e-7)) -
                      tf.math.atan2(b2_wh[..., 0], tf.maximum(b2_wh[..., 1], 1e-7))) / (math.pi * math.pi)
    alpha = v / tf.maximum((1.0 - iou + v), 1e-7)  # 防止当形状与y_true一致时, v为0的情况
    ciou = ciou - alpha * v

    ciou = ciou[..., tf.newaxis]
    ciou = tf.where(tf.math.is_nan(ciou), tf.zeros_like(ciou), ciou)
    return ciou


def yolo_loss(y_true, y_pred, anchors, num_classes, ignore_thresh, label_smoothing=0.1):
    num_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    # input_shape, batch_size
    input_shape = tf.cast(tf.shape(y_pred[0])[1:3] * 32, dtype=y_true[0].dtype)
    batch_size = tf.shape(y_true[0])[0]

    # 初始化3个感受野下的误差
    loss = 0
    for l in range(num_layers):
        # -----------------------------------------------------------#
        #  以第一个特征层(m,13,13,3,5+k)
        #   取出该特征层中存在目标的点的位置。(m,13,13,3,1)
        # -----------------------------------------------------------#
        object_mask = y_true[l][..., 4:5]
        # -----------------------------------------------------------#
        #   取出其对应的种类(m,13,13,3,k)
        # -----------------------------------------------------------#
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = smooth_labels(true_class_probs, label_smoothing)

        # -----------------------------------------------------------#
        #   grid        (13,13,1,2) 网格坐标
        #   raw_pred    (m,13,13,3,k+5) 尚未处理的预测结果
        #   pred_xy     (m,13,13,3,2) 解码后的中心坐标
        #   pred_wh     (m,13,13,3,2) 解码后的宽高坐标
        # -----------------------------------------------------------#
        pred_xy, pred_wh, box_confidence, box_class_probs = yolo_head(y_pred[l],
                                                                      anchors[anchor_mask[l]],
                                                                      num_classes, input_shape)

        # -----------------------------------------------------------#
        #   pred_box是解码后的预测的box的位置
        #   (m,13,13,3,4)
        # -----------------------------------------------------------#
        pred_box = tf.concat([pred_xy, pred_wh], axis=-1)

        # -----------------------------------------------------------#
        #   找到负样本群组
        #   ignore_mask用于提取出作为负样本的特征点
        #   (m,13,13,3)
        # -----------------------------------------------------------#
        object_mask_bool = tf.squeeze(tf.cast(object_mask, dtype=tf.bool), axis=-1)
        ignore_mask = tf.TensorArray(dtype=y_true[0].dtype, size=1, dynamic_size=True)

        # -----------------------------------------------------------#
        #   循环体
        # -----------------------------------------------------------#
        def loop_body(b, ignore_mask):
            # -----------------------------------------------------------#
            #   取n个真实框 n,4
            # -----------------------------------------------------------#
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b])
            # -----------------------------------------------------------#
            #   计算预测框与真实框的iou
            #   pred_box    13,13,3,4 预测框的坐标
            #   true_box    n,4 真实框的坐标
            #   iou         13,13,3,n 预测框和真实框的iou
            # -----------------------------------------------------------#
            iou = box_iou(pred_box[b], true_box)

            # -----------------------------------------------------------#
            #   best_iou    13,13,3 每个特征点与真实框的最大重合程度
            # -----------------------------------------------------------#
            best_iou = tf.reduce_max(iou, axis=-1)

            # -----------------------------------------------------------#
            #   忽略预测结果与真实框非常对应特征点，因为这些框已经比较准了
            #   不适合当作负样本，忽略掉。
            # -----------------------------------------------------------#
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_thresh, dtype=true_box.dtype))
            return b + 1, ignore_mask

        # -----------------------------------------------------------#
        #   循环计算ignore_mask
        # -----------------------------------------------------------#
        _, ignore_mask = tf.while_loop(lambda b, *args: b < batch_size, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()

        # -----------------------------------------------------------#
        #   真实框越大，比重越小，小框的比重更大。
        # -----------------------------------------------------------#
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # -----------------------------------------------------------#
        #   计算Ciou loss
        # -----------------------------------------------------------#
        raw_true_box = y_true[l][..., 0:4]
        ciou = box_ciou(pred_box, raw_true_box)
        ciou_loss = object_mask * box_loss_scale * (1 - ciou)

        # ------------------------------------------------------------------------------#
        #   如果该位置本来有框，那么计算1与置信度的交叉熵
        #   如果该位置本来没有框，那么计算0与置信度的交叉熵
        #   在这其中会忽略一部分样本，这些被忽略的样本满足条件best_iou<ignore_thresh
        #   不适合当作负样本，所以忽略掉。
        # ------------------------------------------------------------------------------#

        pos_conf_loss = binary_crossentropy(object_mask, box_confidence)[object_mask_bool]

        neg_conf_loss = (1 - object_mask[..., -1]) * binary_crossentropy(object_mask, box_confidence)
        neg_conf_loss = neg_conf_loss[tf.cast(ignore_mask, dtype=tf.bool)]

        class_loss = EagerMaskedReducalNLL(mask=object_mask[..., -1], target_num=num_classes)(true_class_probs,
                                                                                              box_class_probs)

        num_pos = tf.maximum(tf.reduce_sum(tf.cast(object_mask, dtype=tf.float32)), 1)

        location_loss = tf.reduce_sum(ciou_loss)
        pos_conf_loss = tf.reduce_sum(pos_conf_loss)
        neg_conf_loss = tf.reduce_sum(neg_conf_loss)

        loss += (location_loss + pos_conf_loss + neg_conf_loss) / num_pos + class_loss

    loss = loss / num_layers

    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.where(tf.greater(loss, tf.constant(10000.)), tf.zeros_like(loss), loss)

    return loss
