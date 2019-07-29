import tensorflow as tf

from .discrminator.network import Discriminator
from .generator.generator import Generator

from .filer import load_GD_weights
from . import config

K = tf.keras.backend
layers = tf.keras.layers
optimizers = tf.keras.optimizers


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


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def multiple_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)


def mean_loss(y_true, y_pred):
    return K.mean(y_pred)


def PG_GAN(G, D, latent_size, label_size, resolution,
           num_channels):

    print("Latent size:")
    print(latent_size)

    print("Label size:")
    print(label_size)

    G_train = tf.keras.Sequential([G, D])
    G_train.cur_lod = G.cur_lod

    shape = D.get_input_shape_at(0)[1:]
    gen_input = tf.keras.Input(shape)
    real_input = tf.keras.Input(shape)
    interpolation = tf.keras.Input(shape)

    sub = tf.keras.Subtract()([D(gen_input), D(real_input)])
    norm = GradNorm()([D(interpolation), interpolation])
    D_train = tf.keras.Model(
        [gen_input, real_input, interpolation], [sub, norm])
    D_train.cur_lod = D.cur_lod

    return G_train, D_train


def build_trainable_model(resume_network,
                          num_channels,
                          resolution,
                          label_size,
                          adam_beta1, adam_beta2, adam_epsilon
                          ):
    G = Generator(num_channels=num_channels,
                  resolution=resolution,
                  label_size=label_size, **config.G)
    D = Discriminator(num_channels=num_channels,
                      resolution=resolution,
                      label_size=label_size, **config.D)
    if resume_network:
        print('Resuming weight from:'+resume_network)
        G, D = load_GD_weights(
            G, D, str(config.result_dir_path / resume_network), True)

    G_train, D_train = PG_GAN(
        G, D, config.G['latent_size'], 0, resolution, num_channels)

    print(G.summary())
    print(D.summary())

    G_opt = optimizers.Adam(lr=0.0, beta_1=adam_beta1,
                            beta_2=adam_beta2, epsilon=adam_epsilon)
    D_opt = optimizers.Adam(lr=0.0, beta_1=adam_beta1,
                            beta_2=adam_beta2, epsilon=adam_epsilon)

    if config.loss['type'] == 'wass':
        G_loss = wasserstein_loss
        D_loss = wasserstein_loss
    elif config.loss['type'] == 'iwass':
        G_loss = multiple_loss
        D_loss = [mean_loss, 'mse']
        D_loss_weight = [1.0, config.loss['iwass_lambda']]

    G.compile(G_opt, loss=G_loss)
    D.trainable = False
    G_train.compile(G_opt, loss=G_loss)
    D.trainable = True
    D_train.compile(D_opt, loss=D_loss, loss_weights=D_loss_weight)

    return G, G_train, D, D_train
