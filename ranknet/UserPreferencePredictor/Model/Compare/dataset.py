import tensorflow as tf
from TrainDataGenerator.TFRecordsMaker.util \
    import IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH
from pathlib import Path

SCOPE = 'ranknet_dataset'


def make_dataset(dataset_file_path: str, batch_size: int, name: str):
    dataset_file_path = Path(dataset_file_path)
    if not dataset_file_path.exists():
        raise FileNotFoundError

    with tf.name_scope(f'{name}_{SCOPE}'):
        dataset = \
            tf.data.TFRecordDataset(str(dataset_file_path)) \
            .map(_parse_function) \
            .map(_read_image) \
            .shuffle(batch_size) \
            .batch(batch_size) \
            .repeat()

    return dataset


def _parse_function(example_proto):
    features = {
        'label': tf.io.FixedLenFeature((), tf.int64,
                                       default_value=0),
        'left_image': tf.io.FixedLenFeature((), tf.string,
                                            default_value=""),
        'right_image': tf.io.FixedLenFeature((), tf.string,
                                             default_value=""),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)

    return parsed_features


def _read_image(parsed_features):
    left_image_raw = \
        tf.decode_raw(parsed_features['left_image'], tf.uint8)
    right_image_raw =\
        tf.decode_raw(parsed_features['right_image'], tf.uint8)

    label = tf.cast(parsed_features['label'], tf.int32, name='label')

    float_left_image_raw = tf.cast(left_image_raw, tf.float32)/255
    float_right_image_raw = tf.cast(right_image_raw, tf.float32)/255

    shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]
    left_image = \
        tf.reshape(float_left_image_raw, shape, name='left_image')
    right_image = \
        tf.reshape(float_right_image_raw, shape, name='right_image')

    return ((left_image, right_image), label)
