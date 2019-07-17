import tensorflow as tf
import disc_layers
import common_layers

layers = tf.keras.layers


class ConvBlock:
    def __init__(self,
                 activation,
                 initializer,
                 epsilon,
                 gdrop_strength,
                 use_wscale,
                 use_layernorm,
                 use_gdrop
                 ):

        self.activation = activation
        self.initializer = initializer
        self.epsilon = epsilon
        self.gdrop_strength = gdrop_strength

        self.use_wscale = use_wscale
        self.use_layernorm = use_layernorm
        self.use_gdrop = False  # Todo

    def _GD(self, incoming, name):
        if self.use_gdrop:
            incoming = \
                disc_layers.GDropLayer(name=name+'gd', mode='prop',
                                       strength=self.gdrop_strength)(incoming)

        return incoming

    def __call__(self,
                 net,
                 num_filter,
                 filter_size,
                 pad,
                 name=None):
        with tf.variable_scope(f'conv_block_{name}'):
            net = self._GD(net, name)

            if pad == 'full':
                pad = filter_size - 1
            net = layers.ZeroPadding2D(pad, name=name + 'Pad')(net)

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

            layer = Conv

            if self.use_wscale:
                layer = common_layers.WScaleLayer(layer, name=name + 'ws')
                net = layer(net)
                net = common_layers.AddBiasLayer()(net)
                net = self.activation(net)
            if self.use_layernorm:
                net = disc_layers.LayerNormLayer(
                    layer, self.epsilon, name=name+'ln')(net)

        return net


class DenseBlock:
    def __init__(self, activation, initializer, use_wscale):
        self.activation = activation
        self.initializer = initializer
        self.use_wscale = use_wscale

    def __call__(self,
                 net,
                 size,
                 name=None):
        with tf.variable_scope(f'dense_block_{name}'):
            if self.use_wscale:
                layer = layers.Dense(size, activation=None, use_bias=False,
                                     kernel_initializer=self.initializer,
                                     name=name+'dense')
                net = layer(net)
                net = common_layers.WScaleLayer(layer, name=name+'ws')(net)
                net = common_layers.AddBiasLayer()(net)
                net = self.activation(net)
            else:
                layer = layers.Dense(size, activation=self.activation,
                                     kernel_initializer=self.initializer,
                                     name=name+'dense')
                net = layer(net)

        return net
