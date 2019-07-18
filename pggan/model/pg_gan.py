import tensorflow as tf

K = tf.keras.backend
layers = tf.keras.layers


class GradNorm(layers.Layer):
    def __init__(self, **kwargs):
        super(GradNorm, self).__init__(**kwargs)

    def build(self, input_shapes):
        # Create a trainable weight variable for this layer.
        # Be sure to call this somewhere!
        super(GradNorm, self).build(input_shapes)

    def call(self, inputs):
        target, wrt = inputs
        grads = K.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1,
        keepdims=True))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


def PG_GAN(G, D, latent_size, label_size, resolution, num_channels):
    print(f'Latent size: {latent_size}')
    print(f'Label size: {label_size}')

    # inputs = [Input(shape=[latent_size], name='GANlatents')]
    # if label_size:
    #    inputs += [Input(shape=[label_size], name='GANlabels')]

    # fake = G(inputs)
    # GAN_out = D(fake)

    # G_train = Model(inputs = inputs,outputs = [GAN_out],name = "PG_GAN")
    G_train = tf.keras.Sequential([G, D])
    G_train.cur_lod = G.cur_lod

    shape = D.get_input_shape_at(0)[1:]
    gen_input = tf.keras.Input(shape)
    real_input = tf.keras.Input(shape)
    interpolation = tf.keras.Input(shape)

    sub = layers.Subtract()([D(gen_input), D(real_input)])
    norm = GradNorm()([D(interpolation), interpolation])
    D_train = tf.keras.Model([gen_input, real_input, interpolation],
                             [sub, norm])
    D_train.cur_lod = D.cur_lod

    return G_train, D_train
