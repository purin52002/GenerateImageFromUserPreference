import tensorflow as tf
from TrainDataGenerator.TFRecordsMaker.util \
    import IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH


class TrainDataset:
    def __init__(self, graph: tf.Graph, batch_size: int):
        self.batch_size = batch_size
        with graph.as_default():
            with tf.variable_scope('train_data_set'):
                self.file_path_placeholder = \
                    tf.placeholder(tf.string, shape=[None], name='file_path')

                dataset = self._make_dataset()

                iterator = \
                    tf.data.Iterator.from_structure(
                        dataset.output_types, dataset.output_shapes)
                self.image, self.score = \
                    iterator.get_next()
                self.init_op = iterator.make_initializer(dataset)

    def _make_dataset(self):
        dataset = \
            tf.data.TFRecordDataset(self.file_path_placeholder) \
            .map(self._parse_function) \
            .map(self._read_image) \
            .shuffle(self.batch_size) \
            .batch(self.batch_size)
        return dataset

    def _parse_function(self, example_proto):
        features = {
            'image': tf.FixedLenFeature((), tf.string, default_value=""),
            'score': tf.FixedLenFeature((), tf.float32, default_value=0),
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        return parsed_features

    def _read_image(self, parsed_features):
        image_raw = \
            tf.decode_raw(parsed_features['image'], tf.uint8)

        score = parsed_features['score']

        float_image_raw = tf.cast(image_raw, tf.float32)/255

        shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
        image = \
            tf.reshape(float_image_raw, shape, name='image')

        return image, score
