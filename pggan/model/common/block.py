import tensorflow as tf
from . import common_layers

layers = tf.keras.layers


class NINBlock:
    def __init__(self,
                 activation,
                 initializer,
                 use_wscale):
        self.activation = activation
        self.initializer = initializer
        self.use_wscale = use_wscale

    def __call__(self,
                 net,
                 num_channels,
                 name=None):

        with tf.name_scope(f'NIN_block_{name}'):
            if self.use_wscale:
                NINlayer = layers.Conv2D(num_channels, 1, padding='same',
                                         activation=None, use_bias=False,
                                         kernel_initializer=self.initializer,
                                         name=name+'NIN')

                net = NINlayer(net)
                net = common_layers.WScaleLayer(
                    NINlayer, name=name+'NINWS')(net)

                net = common_layers.AddBiasLayer()(net)
                net = self.activation(net)
            else:
                NINlayer = layers.Conv2D(num_channels, 1, padding='same',
                                         activation=self.activation,
                                         kernel_initializer=self.initializer,
                                         name=name+'NIN')
                net = NINlayer(net)

        return net
