import numpy as np
from PIL import Image, ImageFont, ImageDraw
import configure.config as cfg
from network.yolov4 import YOLOV4
import tensorflow.keras as keras
import tensorflow as tf
from custom.yolo_loss import yolo_loss
from network.post_processing import yolo_eval


class Yolov4_model:
    def __init__(self,
                 anchors: np.ndarray,
                 classes_name: list,
                 learning_rate: float,
                 ignore_thresh:float,
                 score_thresh: float,
                 iou_thresh: float,
                 max_boxes: int):
        '''

        :param anchors:
        :param classes_name:
        :param learning_rate:
        :param score_thresh: 用于在后处理过程中筛选出具有较大confidence的boxes
        :param iou_thresh: 由于非极大值抑制中，从多个框选出一个框，大于iou_thresh则丢弃
        :param max_boxes:
        :param letterbox_image:用于控制图片还原操作，如果使用lettrbox，则对应相关还原操作
        '''

        self.anchors = anchors
        self.classes_name = classes_name
        self.learning_rate = learning_rate
        self.ignore_thresh = ignore_thresh

        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.max_boxes = max_boxes

        self.num_classes = classes_name.__len__()
        self.num_anchors = anchors.__len__()
        # network
        # YOLOV4()中的num_anchors指的是每一层中的anchor数量，self.num_anchors表示总anchor数量
        # 层数等于3，所有除以了3
        self.network = YOLOV4(num_classes=self.num_classes,
                              num_anchors=self.num_anchors // 3)

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)  # 可以使用clipvalue进行梯度剪裁 防止梯度爆炸

        self.train_loss = keras.metrics.Mean()
        self.valid_loss = keras.metrics.Mean()
        self.train_conf = keras.metrics.BinaryAccuracy()
        self.valid_conf = keras.metrics.BinaryAccuracy()

        self.train_class_acc = keras.metrics.CategoricalAccuracy()
        self.valid_class_acc = keras.metrics.CategoricalAccuracy()

    # yolo_loss中有tf.TensorArray()
    # 经过对比，虽然要多次追踪，但依然是静态图要比eager模式运算更快40%左右 原因未知
    @tf.function
    def train(self, sources, targets):
        with tf.GradientTape() as tape:
            logits = self.network(sources)
            loss = yolo_loss(y_true=targets,
                             y_pred=logits,
                             anchors=self.anchors,
                             num_classes=self.num_classes,
                             ignore_thresh=self.ignore_thresh)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        # if not tf.reduce_any([tf.math.is_nan(tf.reduce_sum(gradient)) for gradient in gradients]):
        #     self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        self.train_loss(loss)

        logits = [
            tf.reshape(logit,
                       shape=(tf.shape(logit)[0], tf.shape(logit)[1], tf.shape(logit)[2], self.num_anchors // 3, -1))
            for logit in logits]
        prob_confs = [tf.sigmoid(logit[..., 4:5]) for logit in logits]
        real_confs = [target[..., 4:5] for target in targets]
        [self.train_conf(real_conf, prob_conf) for real_conf, prob_conf in zip(real_confs, prob_confs)]

        object_mask = [tf.squeeze(tf.cast(real_conf, dtype=tf.bool), axis=-1)
                       for real_conf in real_confs]
        prob_classes = [tf.boolean_mask(logit[..., 5:], mask) for logit, mask in zip(logits, object_mask)]
        real_classes = [tf.boolean_mask(target[..., 5:], mask) for target, mask in zip(targets, object_mask)]
        [self.train_class_acc(real_class, prob_class) for real_class, prob_class in
         zip(real_classes, prob_classes)]

    @tf.function
    def validata(self, sources, targets):
        logits = self.network(sources)
        loss = yolo_loss(y_true=targets,
                         y_pred=logits,
                         anchors= self.anchors,
                         num_classes=self.num_classes,
                         ignore_thresh=self.ignore_thresh)
        self.valid_loss(loss)
        logits = [tf.reshape(logit, shape=[tf.shape(logit)[0], tf.shape(logit)[1], tf.shape(logit)[2],
                                           self.num_anchors // 3, -1]) for logit in logits]

        prob_confs = [tf.sigmoid(logit[..., 4:5]) for logit in logits]
        real_confs = [target[..., 4:5] for target in targets]
        [self.valid_conf(real_conf, prob_conf) for real_conf, prob_conf
         in zip(real_confs, prob_confs)]

        object_masks = [tf.squeeze(tf.cast(real_conf, dtype=tf.bool), axis=-1)
                        for real_conf in real_confs]
        prob_classes = [tf.boolean_mask(logit[..., 5:], mask)
                        for logit, mask in zip(logits, object_masks)]
        real_classes = [tf.boolean_mask(target[..., 5:], mask)
                        for target, mask in zip(targets, object_masks)]
        [self.valid_class_acc(real_class, prob_class)
         for real_class, prob_class in zip(real_classes, prob_classes)]

    def generate_sample(self, sources, batch, letterbox_image):
        '''
        画出图像和方框
        :param sources:
        :param batch:
        :return:
        '''
        logits = self.network(sources)
        image_size = tf.shape(sources)[1:3]
        batch_boxes, batch_scores, batch_classes = yolo_eval(logits,
                                                             self.anchors,
                                                             self.num_classes,
                                                             image_size,
                                                             self.max_boxes,
                                                             self.score_thresh,
                                                             self.iou_thresh,
                                                             letterbox_image)
        batch_boxes = [boxes.numpy() for boxes in batch_boxes]
        batch_scores = [scores.numpy() for scores in batch_scores]
        batch_classes = [classes.numpy() for classes in batch_classes]

        # 随机从sources中选择一张图片
        index = np.random.choice(np.shape(sources)[0], 1)[0]
        source = sources[index]
        image = Image.fromarray(np.uint8(source * 255))

        for i, coordinate in enumerate(batch_boxes[index].astype('int')):
            left, top = list(reversed(coordinate[:2]))
            right, bottom = list(reversed(coordinate[2:]))

            font = ImageFont.truetype(font=cfg.font_path,
                                      size=np.floor(4e-2 * image.size[1] + 0.5).astype('int32'))
            label = f'{self.classes_name[batch_classes[index][i]]:s}:{batch_scores[index][i]:.2f}'

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label_size = tuple(map(np.int32, label_size))  # 转为int32，防止流溢出
            label = label.encode('utf-8')
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            draw.rectangle((left, top, right, bottom),
                           outline=(0, 0, 255),
                           width=int(2 * 0.5))
            draw.text(text_origin,
                      text=str(label, 'utf-8'),
                      fill=(0, 255, 0),
                      font=font)
            del draw
        image.save(cfg.sample_path.format(batch), quality=95, subsampling=0)
