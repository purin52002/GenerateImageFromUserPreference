import tensorflow as tf
from pathlib import Path

from TrainDataGenerator.TFRecordsMaker.util \
    import DATASET_TYPE_LIST, EXTENSION

from argparse import ArgumentParser


def disp_tfrecords_length(tfrecord_dir_path: str):
    tfrecord_dir_path = Path(tfrecord_dir_path)

    assert tfrecord_dir_path.exists()

    print(f'directory: {tfrecord_dir_path}')
    for dataset_type in DATASET_TYPE_LIST:
        dataset_length = \
            len(list(tf.python_io.tf_record_iterator(
                str(tfrecord_dir_path/(dataset_type+EXTENSION)))))
        print(f'{dataset_type} length: {dataset_length}')


def disp_score(tfrecord_dir_path: str):
    tfrecord_dir_path = Path(tfrecord_dir_path)

    for dataset_type in DATASET_TYPE_LIST:
        print(f'{dataset_type}: ')
        dataset_path = str(tfrecord_dir_path/(dataset_type+EXTENSION))
        for example in tf.io.tf_record_iterator(dataset_path):
            print(tf.train.Example.FromString(example))
            exit()


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-d', '--dataset_dir_path', required=True)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


if __name__ == "__main__":
    args = _get_args()
    disp_score(args.dataset_dir_path)
