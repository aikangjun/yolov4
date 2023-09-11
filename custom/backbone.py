from custom.customlayers import CSPBlock, ConvBNMish
from custom import *


class CSPDarknet53(layers.Layer):
    def __init__(self,
                 **kwargs):
        super(CSPDarknet53, self).__init__(**kwargs)
        self.conv = ConvBNMish(filters=32,
                               kernel_size=(3, 3),
                               strides=(1, 1))
        self.downsample_1 = CSPBlock(filters=128,
                                     kernel_size=(3, 3),
                                     strides=(2, 2),
                                     num_resblock=1)
        self.downsample_2 = CSPBlock(filters=128,
                                     kernel_size=(3, 3),
                                     strides=(2, 2),
                                     num_resblock=2)
        self.downsample_3 = CSPBlock(filters=256,
                                     kernel_size=(3, 3),
                                     strides=(2, 2),
                                     num_resblock=8)
        self.downsample_4 = CSPBlock(filters=512,
                                     kernel_size=(3, 3),
                                     strides=(2, 2),
                                     num_resblock=8)
        self.downsample_5 = CSPBlock(filters=1024,
                                     kernel_size=(3, 3),
                                     strides=(2, 2),
                                     num_resblock=4)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.downsample_1(x)
        x = self.downsample_2(x)
        p3 = self.downsample_3(x)
        p4 = self.downsample_4(p3)
        p5 = self.downsample_5(p4)
        return p3, p4, p5
