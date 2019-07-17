import tensorflow as tf
import numpy as np

import gen_layers
import gen_block
import common_block
import common_func
import common_layers

layers = tf.keras.layers
activations = tf.keras.activations
initializers = tf.keras.initializers
K = tf.keras.backend


def Generator(
        num_channels=1,
        resolution=32,
        label_size=0,
        fmap_base=4096,
        fmap_decay=1.0,
        fmap_max=256,
        latent_size=None,
        normalize_latents=True,
        use_wscale=True,
        use_pixelnorm=True,
        use_leakyrelu=True,
        use_batchnorm=False,
        tanh_at_end=None,
        **kwargs):
    with tf.name_scope('generator'):
        R = int(np.log2(resolution))

        if not resolution == 2 ** R:
            raise ValueError

        if not resolution >= 4:
            raise ValueError

        cur_lod = K.variable(np.float32(0.0), dtype='float32', name='cur_lod')

        numf = common_func.FeatureNumber(fmap_base, fmap_decay, fmap_max)

        if latent_size is None:
            latent_size = numf(0)

        (act, act_init) = (None, None)
        if use_leakyrelu:
            (act, act_init) = (layers.LeakyReLU(), initializers.he_normal())
        else:
            (act, act_init) = (activations.relu, initializers.he_normal())

        G_convblock = gen_block.ConvBlock(act, act_init, use_wscale,
                                          use_batchnorm,
                                          use_pixelnorm)
        NINblock = common_block.NINBlock(activations.linear,
                                         initializers.he_normal(), use_wscale)

        inputs = [tf.keras.Input(shape=[latent_size], name='Glatents')]
        net = inputs[-1]

        if normalize_latents:
            net = gen_layers.PixelNormLayer(name='Gnorm')(net)
        if label_size:
            inputs += [tf.keras.Input(shape=[label_size], name='Glabels')]
            net = layers.Concatenate(name='G1na')([net, inputs[-1]])
        net = layers.Reshape((1, 1, K.int_shape(net)[1]), name='G1nb')(net)

        net = G_convblock(net, numf(1), 4,  pad='full', name='G1a')
        net = G_convblock(net, numf(1), 3,  pad=1, name='G1b')
        lods = [net]
        for I in range(2, R):
            net = layers.UpSampling2D(2, name='G%dup' % I)(net)
            net = G_convblock(net, numf(I), 3,  pad=1,  name='G%da' % I)
            net = G_convblock(net, numf(I), 3, pad=1,  name='G%db' % I)
            lods += [net]

        lods = [NINblock(l, num_channels, name='Glod%d' % i)
                for i, l in enumerate(reversed(lods))]
        output = common_layers.LODSelectLayer(cur_lod, name='Glod')(lods)
        if tanh_at_end is not None:
            output = activations.Activation('tanh', name='Gtanh')(output)
            if tanh_at_end != 1.0:
                output = layers.Lambda(
                    lambda x: x * tanh_at_end, name='Gtanhs')

        model = tf.keras.Model(inputs=inputs, outputs=[output])
        model.cur_lod = cur_lod
    return model


if __name__ == "__main__":
    from pathlib import Path
    import config
    generator = Generator(**config.Generator)

    summary_dir = Path(__file__).parent/'summary'
    tf.summary.FileWriter(summary_dir, tf.get_default_graph())
