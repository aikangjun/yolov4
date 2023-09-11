from custom import *

class Head(layers.Layer):
    def __init__(self,
                 num_classes: int,
                 num_anchors: int,
                 **kwargs):
        super(Head, self).__init__(**kwargs)
        self.conv_1 = layers.Conv2D(filters=int((5 + num_classes) * num_anchors),
                                    kernel_size=(1, 1),
                                    strides=(1, 1),
                                    padding="SAME")
        self.conv_2 = layers.Conv2D(filters=int((5 + num_classes) * num_anchors),
                                    kernel_size=(1, 1),
                                    strides=(1, 1),
                                    padding="SAME")
        self.conv_3 = layers.Conv2D(filters=int((5 + num_classes) * num_anchors),
                                    kernel_size=(1, 1),
                                    strides=(1, 1),
                                    padding="SAME")

    def call(self, inputs, *args, **kwargs):
        p3, p4, p5 = inputs
        p3 = self.conv_1(p3)
        p4 = self.conv_2(p4)
        p5 = self.conv_3(p5)
        return p3, p4, p5
