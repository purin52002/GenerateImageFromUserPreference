import tensorflow as tf

layers = tf.keras.layers


class EvaluateNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = layers.Conv2D(filters=32, kernel_size=5,
                                      padding='same', activation='relu')

        self.pool2d_1 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv2d_2 = layers.Conv2D(filters=64, kernel_size=5,
                                      padding='same', activation='relu')

        self.pool2d_2 = layers.MaxPool2D(pool_size=2, strides=2)

        self.flatten = layers.Flatten()

        self.dense_1 = layers.Dense(units=1024, activation='relu')

        self.dropout = layers.Dropout(rate=0.5)

        self.dense_2 = layers.Dense(units=1)

    def call(self, inputs, training=False):
        x = self.conv2d_1(inputs)
        x = self.pool2d_1(x)
        x = self.conv2d_2(x)
        x = self.pool2d_2(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        if training:
            x = self.dropout(x, training)
        return self.dense_2(x)


if __name__ == "__main__":
    net = EvaluateNetwork()

    input_shape = (32, 32, 3)

    left_input = tf.keras.Input(shape=input_shape)
    right_input = tf.keras.Input(shape=input_shape)

    x = net(left_input)
    y = net(right_input)

    combined = tf.keras.Model(inputs=[left_input, right_input], outputs=[x, y])
    combined.summary()
