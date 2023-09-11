from custom.backbone import CSPDarknet53
from custom.neck import Neck
from custom.head import Head
from custom import *
import tensorflow as tf


class YOLOV4(models.Model):
    def __init__(self,
                 num_classes: int,
                 num_anchors: int,
                 **kwargs):
        super(YOLOV4, self).__init__(**kwargs)
        self.backbone = CSPDarknet53()
        self.neck = Neck()
        self.head = Head(num_classes=num_classes,
                         num_anchors=num_anchors)

    def call(self, inputs, *args, **kwargs):
        p3, p4, p5 = self.backbone(inputs)
        p3, p4, p5 = self.neck([p3, p4, p5])
        p3, p4, p5 = self.head([p3, p4, p5])
        return p5, p4, p3


if __name__ == '__main__':

    inputs = tf.random.normal(shape=(4, 416, 416, 3))
    yolo = YOLOV4(num_classes=3,
                  num_anchors=3)
    p5, p4, p3 = yolo(inputs)
    1
