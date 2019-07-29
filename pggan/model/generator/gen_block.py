import tensorflow as tf
from ..common import layers as common_layers
from . import gen_layers

layers = tf.keras.layers


class ConvBlock:
    def __init__(self,
                 activation,
                 initializer,
                 use_wscale,
                 use_batchnorm,
                 use_pixelnorm):

        self.activation = activation
        self.initializer = initializer
        self.use_wscale = use_wscale
        self.use_batchnorm = use_batchnorm
        self.use_pixelnorm = use_pixelnorm

    def __call__(self,
                 net,
                 num_filter,
                 filter_size,
                 pad='same',
                 name=None):
        with tf.name_scope(f'conv_block_{name}'):
            if pad == 'full':
                pad = filter_size - 1

            net = layers.ZeroPadding2D(pad, name=name+'Pad')(net)

            if self.use_wscale:
                Conv = layers.Conv2D(num_filter, filter_size, padding='valid',
                                     activation=None,
                                     kernel_initializer=self.initializer,
                                     use_bias=False, name=name+'conv')
            else:
                Conv = layers.Conv2D(num_filter, filter_size, padding='valid',
                                     activation=self.activation,
                                     kernel_initializer=self.initializer,
                                     name=name+'conv')

            net = Conv(net)

            if self.use_wscale:
                net = common_layers.WScaleLayer(Conv, name=name+'WS')(net)
                net = common_layers.AddBiasLayer()(net)
                net = self.activation(net)

            if self.use_batchnorm:
                net = layers.BatchNormalization(name=name+'BN')(net)

            if self.use_pixelnorm:
                net = gen_layers.PixelNormLayer(name=name+'PN')(net)

        return net
