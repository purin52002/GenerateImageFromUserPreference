import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from UserPreferencePredictor.Model.Regression.dataset \
    import TrainDataset, IMAGE_HEIGHT, IMAGE_WIDTH


class ModelBuilder():
    def __init__(
            self, batch_size: int, is_tensor_verbose=False):
        self.is_tensor_verbose = is_tensor_verbose

        with tf.Graph().as_default() as graph:
            self.train_dataset = TrainDataset(graph, batch_size)
            with tf.variable_scope('predict_model'):
                self._build_model()

            self.merged_summary = tf.summary.merge_all()
            self.global_variables_init_op = tf.global_variables_initializer()

            self.saver = tf.train.Saver()
            self.sess = tf.Session(graph=graph)

    def _build_evaluate_network(self, input_layer: tf.Tensor):
        array_height, array_width = IMAGE_HEIGHT, IMAGE_WIDTH

        conv1_filter_num = 32
        conv1_layer = \
            tf.layers.conv2d(
                inputs=input_layer, filters=conv1_filter_num, kernel_size=5,
                padding='same', activation=tf.nn.relu, name='conv1_layer')

        pooling1_layer = \
            tf.layers.max_pooling2d(
                inputs=conv1_layer, pool_size=2, strides=2,
                name='pooling1_layer')
        array_height, array_width = array_height//2, array_width//2

        conv2_filter_num = conv1_filter_num*2
        conv2_layer = \
            tf.layers.conv2d(
                inputs=pooling1_layer, filters=conv2_filter_num, kernel_size=5,
                padding='same', activation=tf.nn.relu, name='conv2_layer')

        pooling2_layer = \
            tf.layers.max_pooling2d(
                inputs=conv2_layer, pool_size=2, strides=2,
                name='pooling2_layer')
        array_height, array_width = array_height//2, array_width//2

        flatten_layer = \
            tf.reshape(
                pooling2_layer,
                shape=[-1, array_height*array_width*conv2_filter_num],
                name='flatten_layer')

        dense_layer = \
            tf.layers.dense(
                inputs=flatten_layer, units=1024,
                activation=tf.nn.relu, name='dense_layer')
        dropout_layer = \
            tf.layers.dropout(
                inputs=dense_layer,
                rate=self.dropout_placeholder, name='dropout_layer')

        output_layer = tf.layers.dense(
            dropout_layer, units=1, activation=None, name='output_layer')

        if self.is_tensor_verbose:
            print('--- evaluate network ---')
            print(conv1_layer)
            print(pooling1_layer)
            print(conv2_layer)
            print(pooling2_layer)
            print(flatten_layer)
            print(dense_layer)
            print(dropout_layer)
            print(output_layer)
            print('')

        return output_layer

    def _build_loss_func(self, evaluate: tf.Tensor):

        self.loss_op = \
            tf.losses.mean_squared_error(
                labels=self.train_dataset.score, predictions=evaluate)
        tf.summary.scalar('loss', self.loss_op)

        global_step = tf.train.get_or_create_global_step()
        self.train_op = \
            tf.train.AdamOptimizer() \
            .minimize(self.loss_op, global_step=global_step)

        if self.is_tensor_verbose:
            print('--- loss func ---')
            print(evaluate)
            print(self.loss_op)
            print('')

    def _build_model(self):
        with tf.variable_scope('placeholder'):
            self.dropout_placeholder = tf.placeholder(tf.float32)

        with tf.variable_scope('evaluate_network'):
            self.evaluate_net = \
                self._build_evaluate_network(self.train_dataset.image)

        with tf.variable_scope('evaluate'):
            evaluate = \
                tf.reduce_sum(
                    self.evaluate_net, axis=1, name='evaluate')

        with tf.variable_scope('loss_function'):
            self._build_loss_func(evaluate)

    def _image_to_array(self, image: Image):
        resized_image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        return np.asarray(resized_image).astype(np.float32)/255

    def restore(self, check_point_path: str):
        try:
            self.saver.restore(
                self.sess, str(Path(check_point_path).joinpath('save')))
            return True
        except ValueError:
            return False


if __name__ == '__main__':
    load_dir = str(Path(__file__).parent/'predict_model_test')
    print(ModelBuilder(1, True).restore(load_dir))
