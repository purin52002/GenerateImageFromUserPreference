import tensorflow as tf
import numpy as np

import common_block
import common_layers
import common_func

import disc_block
import disc_layers

layers = tf.keras.layers
initializers = tf.keras.initializers
activations = tf.keras.activations
K = tf.keras.backend


def Discriminator(
        num_channels=1,        # Overridden based on dataset.
        resolution=32,       # Overridden based on dataset.
        label_size=0,        # Overridden based on dataset.
        fmap_base=4096,
        fmap_decay=1.0,
        fmap_max=256,
        mbstat_func='Tstdeps',
        mbstat_avg='all',
        mbdisc_kernels=None,
        use_wscale=True,
        use_gdrop=True,
        use_layernorm=False,
        **kwargs):

    with tf.name_scope('discriminator'):

        epsilon = 0.01
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4
        cur_lod = K.variable(np.float(0.0), dtype='float32', name='cur_lod')
        gdrop_strength = K.variable(
            np.float(0.0), dtype='float32', name='gdrop_strength')

        numf = common_func.FeatureNumber(fmap_base, fmap_decay, fmap_max)

        NINblock = common_block.NINBlock(layers.LeakyReLU(),
                                         initializers.he_normal(), use_wscale)

        def Downscale2DLayer(incoming, scale_factor, **kwargs):
            pool2d = layers.AveragePooling2D(pool_size=scale_factor, **kwargs)
            return pool2d(incoming)

        ConvBlock = disc_block.ConvBlock(layers.LeakyReLU(),
                                         initializers.he_normal(),
                                         epsilon,
                                         gdrop_strength,
                                         use_wscale,
                                         use_layernorm,
                                         use_gdrop)

        DenseBlock = disc_block.DenseBlock(layers.LeakyReLU(),
                                           initializers.he_normal(),
                                           use_wscale)

        inputs = layers.Input(shape=[2**R, 2**R, num_channels], name='Dimages')
        net = NINblock(inputs, numf(R-1), name='D%dx' % (R-1))

        for i in range(R-1, 1, -1):
            net = ConvBlock(net, numf(i), 3,  1, name='D%db' % i)
            net = ConvBlock(net, numf(i - 1), 3,  1, name='D%da' % i)
            net = Downscale2DLayer(net, name='D%ddn' % i, scale_factor=2)
            lod = Downscale2DLayer(inputs, name='D%dxs' %
                                   (i - 1), scale_factor=2 ** (R - i))
            lod = NINblock(lod, numf(i - 1),  name='D%dx' % (i - 1))

            select_layer = \
                common_layers.LODSelectLayer(cur_lod, name='D%dlod' %
                                             (i - 1),
                                             first_incoming_lod=R - i - 1)
            net = select_layer([net, lod])

        if mbstat_avg is not None:
            net = disc_layers.MinibatchStatConcatLayer(
                averaging=mbstat_avg, name='Dstat')(net)

        if mbdisc_kernels:
            net = disc_layers.MinibatchLayer(mbdisc_kernels, name='Dmd')(net)

        net = ConvBlock(net, numf(1), 3,  1, name='D1b')
        net = ConvBlock(net, numf(0), 4,  0, name='D1a')

        net = DenseBlock(net, 1, name='Dscores')
        output_layers = [net]
        if label_size:
            output_layers += [DenseBlock(net, label_size,  name='Dlabels')]

        model = tf.keras.Model(inputs=[inputs], outputs=output_layers)
        model.cur_lod = cur_lod
        model.gdrop_strength = gdrop_strength
    return model


if __name__ == "__main__":
    from pathlib import Path
    discriminator = Discriminator()

    summary_dir = Path(__file__).parent/'summary'
    tf.summary.FileWriter(summary_dir, tf.get_default_graph())
