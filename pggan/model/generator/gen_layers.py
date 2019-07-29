import tensorflow as tf

layers = tf.keras.layers
K = tf.keras.backend


class PixelNormLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(PixelNormLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs**2, axis=-1, keepdims=True) + 1.0e-8
        return inputs / K.sqrt(mean)

    def compute_output_shape(self, input_shape):
        return input_shape


