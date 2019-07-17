import tensorflow as tf
import numpy as np

layers = tf.keras.layers
K = tf.keras.backend


class WScaleLayer(layers.Layer):
    def __init__(self, incoming, activation=None, **kwargs):
        self.incoming = incoming
        self.activation = tf.keras.activations.get(activation)
        super(WScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel = K.get_value(self.incoming.kernel)
        scale = np.sqrt(np.mean(kernel ** 2))
        K.set_value(self.incoming.kernel, kernel/scale)
        self.scale = self.add_weight(
            name='scale', shape=scale.shape, trainable=False,
            initializer='zeros')
        K.set_value(self.scale, scale)
        super(WScaleLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = inputs * self.scale
        return self.activation(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class AddBiasLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AddBiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(name='bias',
                                    shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True)
        super(AddBiasLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.bias is not None:
            inputs = K.bias_add(inputs, self.bias)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class ACTVResizeLayer(layers.Layer):
    def __init__(self, si, so, **kwargs):
        self.si = si
        self.so = so
        super(ACTVResizeLayer, self).__init__(**kwargs)

    def call(self, v, **kwargs):
        assert len(self.si) == len(self.so) and self.si[0] == self.so[0]

        # Decrease feature maps.  Attention: channels last
        if self.si[-1] > self.so[-1]:
            v = v[..., :self.so[-1]]

        # Increase feature maps.  Attention:channels last
        if self.si[-1] < self.so[-1]:
            z = K.zeros(
                (self.so[:-1] + (self.so[-1] - self.si[-1])), dtype=v.dtype)
            v = K.concatenate([v, z])

        # Shrink spatial axis
        if len(self.si) == 4 and \
                (self.si[1] > self.so[1] or self.si[2] > self.so[2]):
            divisible_1 = self.si[1] % self.so[1] == 0
            divisible_2 = self.si[2] % self.so[2] == 0
            assert divisible_1 and divisible_2
            pool_size = (self.si[1] / self.so[1], self.si[2] / self.so[2])
            strides = pool_size
            v = K.pool2d(v, pool_size=pool_size, strides=strides,
                         padding='same', data_format='channels_last',
                         pool_mode='avg')

        # Extend spatial axis
        for i in range(1, len(self.si) - 1):
            if self.si[i] < self.so[i]:
                assert self.so[i] % self.si[i] == 0
                v = K.repeat_elements(v, rep=int(
                    self.so[i] / self.si[i]), axis=i)

        return v

    def compute_output_shape(self, input_shape):
        return self.so


class LODSelectLayer(layers.Layer):
    def __init__(self, cur_lod, first_incoming_lod=0, ref_idx=0, min_lod=None,
                 max_lod=None, **kwargs):
        super(LODSelectLayer, self).__init__(**kwargs)
        self.cur_lod = cur_lod
        self.first_incoming_lod = first_incoming_lod
        self.ref_idx = ref_idx
        self.min_lod = min_lod
        self.max_lod = max_lod

    def call(self, inputs):
        self.input_shapes = [K.int_shape(input) for input in inputs]
        v = [ACTVResizeLayer(K.int_shape(input),
                             self.input_shapes[self.ref_idx])(input)
             for input in inputs]
        lo = np.clip(int(np.floor(self.min_lod - self.first_incoming_lod)),
                     0, len(v)-1) if self.min_lod is not None else 0
        hi = np.clip(int(np.ceil(self.max_lod - self.first_incoming_lod)),
                     lo, len(v)-1) if self.max_lod is not None else len(v)-1
        t = self.cur_lod - self.first_incoming_lod
        r = v[hi]
        for i in range(hi-1, lo-1, -1):  # i = hi-1, hi-2, ..., lo
            r = K.switch(K.less(t, i+1), v[i] * ((i+1)-t) + v[i+1] * (t-i), r)
        if lo < hi:
            r = K.switch(K.less_equal(t, lo), v[lo], r)
        return r

    def compute_output_shape(self, input_shape):
        return self.input_shapes[self.ref_idx]
