from PIL import Image
import numpy as np
import tensorflow as tf
from pathlib import Path
from exception import TFRecordsMakerException


class RecordException(TFRecordsMakerException):
    pass


def _make_label(left_score: float, right_score: float):
    if left_score > right_score:
        return 0

    elif right_score > left_score:
        return 1

    else:
        raise RecordException('score is same')


def _make_record(left_image: Image.Image, right_image: Image.Image,
                 label: int):
    left_array = np.asarray(left_image)
    right_array = np.asarray(right_image)

    features = \
        tf.train.Features(
            feature={
                'label':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label])),
                    'left_image':
                    tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[left_array.tobytes()])),
                    'right_image':
                    tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[right_array.tobytes()]))
            }
        )
    example = tf.train.Example(features=features)
    record = example.SerializeToString()
    return record


def get_record_yield(parsed_info: dict):
    left_dir_path, right_dir_path = parsed_info['param']
    left_score, right_score = parsed_info['score']

    for left_image_path in Path(left_dir_path).iterdir():
        for right_image_path in Path(right_dir_path).iterdir():
            try:
                label = _make_label(left_score, right_score)

                left_image = Image.open(str(left_image_path))
                right_image = Image.open(str(right_image_path))

                record = _make_record(left_image, right_image, label)
                yield record
            except RecordException:
                pass
