from custom import *
from custom.customlayers import SPPBlock, ConvBNLeaky, ConvSet
import tensorflow as tf


class Neck(layers.Layer):
    def __init__(self,
                 **kwargs):
        super(Neck, self).__init__(**kwargs)

        self.conv_1 = ConvBNLeaky(filters=128,
                                  kernel_size=(1, 1),
                                  strides=(1, 1))
        self.conv_2 = ConvBNLeaky(filters=256,
                                  kernel_size=(1, 1),
                                  strides=(1, 1))
        self.convset_1 = ConvSet(filters=512)
        self.spp = SPPBlock()
        self.convset_2 = ConvSet(filters=512)

        self.conv_3 = ConvBNLeaky(filters=256,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  upsample=True)
        self.convset_3 = ConvSet(filters=256)
        self.conv_4 = ConvBNLeaky(filters=128,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  upsample=True)
        self.convset_4 = ConvSet(filters=128)

        self.conv_5 = ConvBNLeaky(filters=256,
                                  kernel_size=(3, 3),
                                  strides=(2, 2))
        self.convset_5 = ConvSet(filters=256)
        self.conv_6 = ConvBNLeaky(filters=512,
                                  kernel_size=(3, 3),
                                  strides=(2, 2))
        self.convset_6 = ConvSet(filters=256)

        self.conv_7 = ConvBNLeaky(filters=256,
                                  kernel_size=(3, 3),
                                  strides=(1, 1))
        self.conv_8 = ConvBNLeaky(filters=512,
                                  kernel_size=(3, 3),
                                  strides=(1, 1))
        self.conv_9 = ConvBNLeaky(filters=1024,
                                  kernel_size=(3, 3),
                                  strides=(1, 1))

    def call(self, inputs, *args, **kwargs):
        p3, p4, p5 = inputs

        p3 = self.conv_1(p3)
        p4 = self.conv_2(p4)
        p5 = self.convset_1(p5)
        p5 = self.spp(p5)
        p5 = self.convset_2(p5)  #

        p5_upsample = self.conv_3(p5)
        p4 = tf.concat([p4, p5_upsample], axis=-1)
        p4 = self.convset_3(p4)  #
        p4_upsample = self.conv_4(p4)
        p3 = tf.concat([p3, p4_upsample], axis=-1)
        p3 = self.convset_4(p3)  #

        p3_downsample = self.conv_5(p3)
        p4 = tf.concat([p4, p3_downsample], axis=-1)
        p4 = self.convset_5(p4)  #
        p4_downsample = self.conv_6(p4)
        p5 = tf.concat([p5, p4_downsample], axis=-1)
        p5 = self.convset_6(p5)  #

        p3 = self.conv_7(p3)
        p4 = self.conv_8(p4)
        p5 = self.conv_9(p5)
        return p3, p4, p5
