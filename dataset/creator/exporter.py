from pathlib import Path
import numpy as np
import tensorflow as tf
from exception import DatasetCreatorException


class ExporterException(DatasetCreatorException):
    pass


class TFRecordExporter:
    def __init__(self, tfrecord_dir_path: str, expected_images,
                 print_progress=True, progress_interval=10):
        self.tfrecord_dir_path = Path(tfrecord_dir_path)
        self.tfr_prefix = \
            str(self.tfrecord_dir_path/self.tfrecord_dir_path.name)
        self.expected_images = expected_images
        self.cur_images = 0
        self.shape = None
        self.resolution_log2 = None
        self.tfr_writers = []
        self.print_progress = print_progress
        self.progress_interval = progress_interval

        if self.print_progress:
            print(f'Creating dataset {tfrecord_dir_path}')

        self.tfrecord_dir_path.mkdir(parents=True, exist_ok=True)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    # Note: Images and labels must be added in shuffled order.
    def choose_shuffled_order(self):
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):
        is_over_interval = self.cur_images % self.progress_interval == 0
        if self.print_progress and is_over_interval:
            print('%d / %d\r' %
                  (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))

            if self.shape[0] not in [1, 3]:
                msg = 'required channel: 1 or 3. '\
                    f'input channel: {self.shape[0]}'

                raise ExporterException(msg)

            if not self.shape[1] == self.shape[2]:
                raise ExporterException

            if not self.shape[1] == 2**self.resolution_log2:
                raise ExporterException

            tfr_opt = tf.io.TFRecordOptions(
                tf.python_io.TFRecordCompressionType.NONE)

            for lod in range(self.resolution_log2 - 1):
                tfr_file = self.tfr_prefix + \
                    '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(
                    tf.io.TFRecordWriter(tfr_file, tfr_opt))

        if not img.shape == self.shape:
            raise ExporterException

        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] +
                       img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
