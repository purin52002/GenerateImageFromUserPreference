import tensorflow as tf
import numpy as np

layers = tf.keras.layers
K = tf.keras.backend
activations = tf.keras.activations


class GDropLayer(layers.Layer):
    def __init__(self, mode='mul', strength=0.4, axes=(0, 3), normalize=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert mode in ('drop', 'mul', 'prop')

        self.mode = mode
        self.strength = strength
        self.axes = [axes] if isinstance(axes, int) else list(axes)
        self.normalize = normalize  # If true, retain overall signal variance.
        self.gain = None      # For experimentation.

    def build(self, input_shape):
        self._input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs, deterministic=False, **kwargs):
        if self.gain is not None:
            inputs = inputs * self.gain
        if deterministic or self.strength is None:
            return inputs

        in_shape = self._input_shape
        in_axes = range(len(in_shape))
        in_shape = [in_shape[axis] if in_shape[axis] is not None
                    else inputs.shape[axis]
                    for axis in in_axes]  # None => Theano expr
        rnd_shape = [in_shape[axis].value for axis in self.axes]
        broadcast = [self.axes.index(axis) if axis in self.axes else 'x'
                     for axis in in_axes]
        print(in_axes)
        print(self.axes)
        print(broadcast)
        exit()
        one = K.constant(1)

        print(rnd_shape)

        if self.mode == 'drop':
            p = one - self.strength
            rnd = K.random_binomial(
                tuple(rnd_shape), p=p, dtype=inputs.dtype) / p

        elif self.mode == 'mul':
            rnd = (
                one + self.strength) ** K.random_normal(tuple(rnd_shape),
                                                        dtype=inputs.dtype)

        elif self.mode == 'prop':
            coef = self.strength * \
                K.constant(np.sqrt(np.float32(self._input_shape[1].value)))
            rnd = K.random_normal(
                tuple(rnd_shape[1:]), dtype=inputs.dtype) * coef + one

        else:
            raise ValueError('Invalid GDropLayer mode', self.mode)

        if self.normalize:
            rnd = rnd / K.sqrt(K.mean(rnd ** 2, axis=3, keepdims=True))
        return inputs * K.permute_dimensions(rnd, broadcast)

    def compute_output_shape(self, input_shape):
        return input_shape


class LayerNormLayer(layers.Layer):
    def __init__(self, incoming, epsilon, **kwargs):
        super(LayerNormLayer, self).__init__(**kwargs)
        self.incoming = incoming
        self.epsilon = epsilon

    def build(self, input_shape):
        gain = np.float32(1.0)
        self.gain = self.add_weight(
            name='gain', shape=gain.shape,  trainable=True,
            initializer='zeros')
        K.set_value(self.gain, gain)
        self.bias = None

        # steal bias
        if hasattr(self.incoming, 'bias') and self.incoming.bias is not None:
            bias = K.get_value(self.incoming.bias)
            self.bias = self.add_param(name='bias', shape=bias.shape)
            K.set_value(self.bias, bias)
            # del self.incoming.params[self.incoming.bias]
            # self.incoming.bias = None
        self.activation = activations.get('linear')

        # steal nonlinearity
        if hasattr(self.incoming, 'activation') and \
                self.incoming.activation is not None:
            self.activation = self.incoming.activation
            self.incoming.activation = activations.get('linear')

        super(LayerNormLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        avg_axes = range(1, len(self.input_shape()))

        # subtract mean
        inputs = inputs - K.mean(inputs, axis=avg_axes,
                                 keepdims=True)

        # divide by stdev
        inputs = inputs * 1.0/K.sqrt(K.mean(K.square(inputs), axis=avg_axes,
                                            keepdims=True) + self.epsilon)

        # multiply by gain
        inputs = inputs * self.gain

        if self.bias is not None:
            inputs = inputs + \
                K.expand_dims(K.expand_dims(K.expand_dims(self.bias, 0), 0), 0)
        return self.activation(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class MinibatchStatConcatLayer(layers.Layer):
    def __init__(self, averaging='all', **kwargs):
        self.averaging = averaging.lower()
        super(MinibatchStatConcatLayer, self).__init__(**kwargs)
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            valid_list = ['all', 'flat', 'spatial', 'none', 'gpool']
            assert self.averaging in valid_list, \
                f'Invalid averaging mode is {self.averaging}'

        self.adjusted_std = lambda x, **kwargs: K.sqrt(
            K.mean((x - K.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def call(self, inputs, **kwargs):
        s = list(K.int_shape(inputs))
        s[0] = tf.shape(inputs)[0]
        # per activation, over minibatch dim
        vals = self.adjusted_std(inputs, axis=0, keepdims=True)

        # average everything --> 1 value per minibatch
        if self.averaging == 'all':
            vals = K.mean(vals, keepdims=True)
            reps = s
            reps[-1] = 1
            reps[0] = tf.shape(inputs)[0]
            vals = K.tile(vals, reps)

        # average spatial locations
        elif self.averaging == 'spatial':
            if len(s) == 4:
                vals = K.mean(vals, axis=(1, 2), keepdims=True)
            reps = s
            reps[-1] = 1
            vals = K.tile(vals, reps)

        # no averaging, pass on all information
        elif self.averaging == 'none':
            vals = K.repeat_elements(vals, rep=s[0], axis=0)

        # EXPERIMENTAL:
        # compute variance (func) over minibatch AND spatial locations.
        elif self.averaging == 'gpool':
            if len(s) == 4:
                vals = self.adjusted_std(inputs, axis=(0, 1, 2), keepdims=True)
            reps = s
            reps[-1] = 1
            vals = K.tile(vals, reps)
        elif self.averaging == 'flat':
            # variance of ALL activations --> 1 value per minibatch
            vals = self.adjusted_std(inputs, keepdims=True)
            reps = s
            reps[-1] = 1
            vals = K.tile(vals, reps)

        # average everything over n groups of feature maps
        # --> n values per minibatch
        elif self.averaging.startswith('group'):
            n = int(self.averaging[len('group'):])
            vals = vals.reshape((1, s[1], s[2], n, s[3]/n))
            vals = K.mean(vals, axis=(1, 2, 4), keepdims=True)
            vals = vals.reshape((1, 1, 1, n))
            reps = s
            reps[-1] = 1
            vals = K.tile(vals, reps)
        else:
            raise ValueError('Invalid averaging mode', self.averaging)
        return K.concatenate([inputs, vals], axis=-1)

    def compute_output_shape(self, input_shape):
        s = list(input_shape)
        if self.averaging == 'all':
            s[-1] += 1
        elif self.averaging == 'flat':
            s[-1] += 1
        elif self.averaging.startswith('group'):
            s[-1] += int(self.averaging[len('group'):])
        else:
            s[-1] *= 2
        return tuple(s)


class MinibatchLayer(layers.Layer):
    def __init__(self, num_kernels, dim_per_kernel=5, theta=None,
                 log_weight_scale=None, b=None, init=False, **kwargs):
        super(MinibatchLayer, self).__init__(**kwargs)
        self.num_kernels = num_kernels
        self.dim_per_kernel = dim_per_kernel
        self.theta_arg = theta
        self.log_weight_scale_arg = log_weight_scale
        self.b_arg = b
        self.init_arg = init

    def build(self, input_shape):
        num_inputs = int(np.prod(input_shape[1:]))
        self.theta = self.add_weight(name='theta',
                                     shape=(num_inputs, self.num_kernels,
                                            self.dim_per_kernel),
                                     initializer='zeros')

        if self.theta_arg is None:
            K.set_value(self.theta,
                        K.random_normal((num_inputs, self.num_kernels,
                                         self.dim_per_kernel),
                                        0.0, 0.05))

        self.log_weight_scale = self.add_weight(name='log_weight_scale',
                                                shape=(self.num_kernels,
                                                       self.dim_per_kernel),
                                                initializer='zeros')

        if self.log_weight_scale_arg is None:
            K.set_value(self.log_weight_scale, K.constant(
                0.0, shape=(self.num_kernels, self.dim_per_kernel)))

        l2 = K.sqrt(K.sum(K.square(self.theta), axis=0))
        self.kernel = self.theta * K.expand_dims(
            K.permute_dimensions(K.exp(self.log_weight_scale)/l2, [0, 1]), 0)

        self.bias = self.add_weight(name='bias', shape=(
            self.num_kernels,), initializer='zeros')

        if self.b_arg is None:
            K.set_value(self.bias, K.constant(-1.0, shape=(self.num_kernels,)))

        super(MinibatchLayer, self).build(input_shape)

    def call(self, inputs, **kargs):
        if K.ndim(inputs) > 2:
            # if the inputs has more than two dimensions, flatten it into a
            # batch of feature vectors.
            inputs = K.flatten(inputs)
        actv = K.batch_dot(inputs, self.kernel, [[1], [0]])

        permute_012 = K.expand_dims(K.permute_dimensions(actv, [0, 1, 2]))
        permute_120 = K.expand_dims(K.permute_dimensions(actv, [1, 2, 0]), 0)

        abs_dif = K.sum(K.abs(permute_012-permute_120), axis=2) + \
            1e6*K.expand_dims(K.eye(K.int_shape(inputs)[0]), 1)

        if self.init_arg:
            mean_min_abs_dif = 0.5 * K.mean(K.min(abs_dif, axis=2), axis=0)
            abs_dif /= K.expand_dims(K.expand_dims(mean_min_abs_dif, 0))
            self.init_updates = [
                (self.log_weight_scale,
                 self.log_weight_scale-K.expand_dims(K.log(mean_min_abs_dif)))]
        f = K.sum(K.exp(-abs_dif), axis=2)

        if self.init_arg:
            mf = K.mean(f, axis=0)
            f -= K.expand_dims(mf, 0)
            self.init_updates += [(self.bias, -mf)]
        else:
            f += K.expand_dims(self.bias, 0)

        return K.concatenate([inputs, f], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:])+self.num_kernels)
