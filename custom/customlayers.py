from custom import *
import tensorflow as tf


class Mish(layers.Layer):
    '''
        Mish Activation Function.
        .. math::
            mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
            tanh=(1 - e^{-2x})/(1 + e^{-2x})
        Shape:
            - Input: Arbitrary. Use the keyword argument `input_shape`
            (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a model.
            - Output: Same shape as the input.
        '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, x, *args, **kwargs):
        x = x * tf.nn.tanh(tf.nn.softplus(x))
        return x

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        '''
         compute_output_shape(self, input_shape)：为了能让Keras内部shape的匹配检查通过，
         这里需要重写compute_output_shape方法去覆盖父类中的同名方法，来保证输出shape是正确的。
         父类Layer中的compute_output_shape方法直接返回的是input_shape这明显是不对的，
         所以需要我们重写这个方法。所以这个方法也是4个要实现的基本方法之一。
        '''
        return input_shape


class ConvBNMish(layers.Layer):
    '''
    原始yolov4中使用Mish激活
    Mish激活函数可能是在训练时，导致梯度消失的原因。考虑替换为ReLU或Leaky ReLU
    '''

    def __init__(self,
                 filters: int,
                 kernel_size: tuple,
                 strides: tuple,
                 padding: str = 'SAME',
                 **kwargs):
        super(ConvBNMish, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  kernel_initializer=keras.initializers.random_normal(),
                                  kernel_regularizer=keras.regularizers.l2())  # 设置kernel_regularizer 与在目标函数设置weight_decay是相同的效果
        self.bn = layers.BatchNormalization()
        self.mish = Mish()

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.mish(x)
        return x


class ResBlock(layers.Layer):
    def __init__(self,
                 filters: int,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.conv_1 = ConvBNMish(filters=filters,
                                 kernel_size=(1, 1),
                                 strides=(1, 1))
        self.conv_2 = ConvBNMish(filters=filters,
                                 kernel_size=(3, 3),
                                 strides=(1, 1))

    def call(self, inputs, *args, **kwargs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        outputs = inputs + x
        return outputs


class CSPBlock(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple,
                 strides: tuple,
                 num_resblock: int,
                 padding: str = 'SAME',
                 **kwargs):
        super(CSPBlock, self).__init__(**kwargs)
        self.conv_1 = ConvBNMish(filters=filters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding)
        self.conv_part1 = ConvBNMish(filters=int(filters // 2),
                                     kernel_size=(1, 1),
                                     strides=(1, 1))
        self.conv_part2 = ConvBNMish(filters=int(filters // 2),
                                     kernel_size=(1, 1),
                                     strides=(1, 1))
        self.res_seq = models.Sequential([ResBlock(filters=int(filters // 2))
                                          for i in range(num_resblock)])
        self.conv_3 = ConvBNMish(filters=int(filters // 2),
                                 kernel_size=(1, 1),
                                 strides=(1, 1))
        self.conv_4 = ConvBNMish(filters=filters,
                                 kernel_size=(1, 1),
                                 strides=(1, 1))

    def call(self, inputs, *args, **kwargs):
        x = self.conv_1(inputs)

        branch_x = self.conv_part1(x)
        main_x = self.conv_part2(x)

        main_x = self.res_seq(main_x)
        main_x = self.conv_3(main_x)

        x = tf.concat([branch_x, main_x], axis=-1)
        x = self.conv_4(x)
        return x


class SPPBlock(layers.Layer):
    def __init__(self,
                 **kwargs):
        super(SPPBlock, self).__init__(**kwargs)
        self.maxpool_1 = layers.MaxPool2D(pool_size=(5, 5),
                                          strides=(1, 1),
                                          padding="SAME")
        self.maxpool_2 = layers.MaxPool2D(pool_size=(9, 9),
                                          strides=(1, 1),
                                          padding="SAME")
        self.maxpool_3 = layers.MaxPool2D(pool_size=(13, 13),
                                          strides=(1, 1),
                                          padding="SAME")

    def call(self, inputs, *args, **kwargs):
        p1 = self.maxpool_1(inputs)
        p2 = self.maxpool_2(inputs)
        p3 = self.maxpool_3(inputs)
        outputs = tf.concat([inputs, p1, p2, p3], axis=-1)
        return outputs


class ConvBNLeaky(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple,
                 strides: tuple,
                 padding: str = "SAME",
                 upsample: bool = False,
                 **kwargs):
        super(ConvBNLeaky, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  kernel_regularizer=keras.regularizers.l2(
                                      1e-2))  # 设置kernel_regularizer 与在目标函数设置weight_decay是相同的效果 防止梯度爆炸
        self.bn = layers.BatchNormalization()
        self.leakyrelu = layers.LeakyReLU(alpha=0.2)

        if upsample:
            self.upsample = layers.UpSampling2D()

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.leakyrelu(x)
        if hasattr(self, "upsample"):
            x = self.upsample(x)
        return x


class ConvSet(layers.Layer):
    def __init__(self,
                 filters: int,
                 **kwargs):
        super(ConvSet, self).__init__(**kwargs)
        if filters == 512:
            self.conv_set = models.Sequential([ConvBNLeaky(filters=filters,
                                                           kernel_size=(1, 1),
                                                           strides=(1, 1)),
                                               ConvBNLeaky(filters=int(2 * filters),
                                                           kernel_size=(3, 3),
                                                           strides=(1, 1)),
                                               ConvBNLeaky(filters=filters,
                                                           kernel_size=(1, 1),
                                                           strides=(1, 1))])
        elif filters == 256 or filters == 128:
            self.conv_set = models.Sequential([ConvBNLeaky(filters=filters,
                                                           kernel_size=(1, 1),
                                                           strides=(1, 1)),
                                               ConvBNLeaky(filters=int(2 * filters),
                                                           kernel_size=(3, 3),
                                                           strides=(1, 1)),
                                               ConvBNLeaky(filters=filters,
                                                           kernel_size=(1, 1),
                                                           strides=(1, 1)),
                                               ConvBNLeaky(filters=int(2 * filters),
                                                           kernel_size=(3, 3),
                                                           strides=(1, 1)),
                                               ConvBNLeaky(filters=filters,
                                                           kernel_size=(1, 1),
                                                           strides=(1, 1))])

    def call(self, inputs, *args, **kwargs):
        if hasattr(self, 'conv_set'):
            x = self.conv_set(inputs)
        return x
