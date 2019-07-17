import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    batch = 2
    height = 3
    width = 4
    channel = 1

    target_shape = (height, width, channel)

    a = np.array([
        [[1], [1], [1], [1]],
        [[2], [2], [2], [2]],
        [[3], [3], [3], [3]],
    ], dtype=np.float32)
    assert target_shape == a.shape
    b = a
    c = np.array([
        [[0]],
        [[0]],
        [[0]],
    ], dtype=np.float32)
    x = np.array([a, b], dtype=np.float32)

    std = tf.math.reduce_std(x, axis=0)
    mean = tf.math.reduce_mean(std)
    mean_tensor = tf.fill([height, 1, 1], mean)
    concat = tf.concat([a, mean_tensor], 1)

    with tf.Session() as sess:
        print(sess.run(std))
        print(sess.run(mean))
        print(sess.run(concat))
