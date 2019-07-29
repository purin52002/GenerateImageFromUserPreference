# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import numpy as np
import tensorflow as tf
from pathlib import Path


class DatasetException(Exception):
    pass


# ----------------------------------------------------------------------------
# Parse individual image from a tfrecords file.


def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])


def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)

# ----------------------------------------------------------------------------
# Dataset class that loads data from tfrecords files.


class TFRecordDataset:
    def __init__(self,
                 # Directory containing a collection of tfrecords files.
                 tfrecord_dir_path,
                 # Dataset resolution, None = autodetect.
                 resolution=None,
                 # Repeat dataset indefinitely.
                 repeat=True,
                 # Shuffle data within specified window (megabytes),
                 # 0 = disable shuffling.
                 shuffle_mb=4096,
                 # Amount of data to prefetch (megabytes),
                 # 0 = disable prefetching.
                 prefetch_mb=2048,
                 # Read buffer size (megabytes).
                 buffer_mb=256,
                 # Number of concurrent threads.
                 num_threads=2):

        self.tfrecord_dir_path = Path(tfrecord_dir_path)
        self.resolution = None
        self.resolution_log2 = None
        self.shape = []        # [channel, height, width]
        self.dtype = 'uint8'
        self.dynamic_range = [0, 255]
        self._tf_minibatch_in = None
        self._tf_datasets = dict()
        self._tf_iterator = None
        self._tf_init_ops = dict()
        self._tf_minibatch_np = None
        self._cur_minibatch = -1
        self._cur_lod = -1

        # List tfrecords files and inspect their shapes.
        if not self.tfrecord_dir_path.is_dir():
            raise DatasetException

        tfr_files = sorted(self.tfrecord_dir_path.glob('*.tfrecords'))

        if len(tfr_files) < 1:
            raise DatasetException

        tfr_shapes = []
        for tfr_file in tfr_files:
            tfr_opt = tf.io.TFRecordOptions(
                tf.io.TFRecordCompressionType.NONE)
            for record in tf.io.tf_record_iterator(tfr_file, tfr_opt):
                tfr_shapes.append(parse_tfrecord_np(record).shape)
                break

        # Determine shape and resolution.
        max_shape = max(tfr_shapes, key=lambda shape: np.prod(shape))
        self.resolution = \
            resolution if resolution is not None else max_shape[1]
        self.resolution_log2 = int(np.log2(self.resolution))
        self.shape = [max_shape[0], self.resolution, self.resolution]
        tfr_lods = [self.resolution_log2 -
                    int(np.log2(shape[1])) for shape in tfr_shapes]
        assert all(shape[0] == max_shape[0] for shape in tfr_shapes)
        assert all(shape[1] == shape[2] for shape in tfr_shapes)
        assert all(shape[1] == self.resolution // (2**lod)
                   for shape, lod in zip(tfr_shapes, tfr_lods))
        assert all(lod in tfr_lods for lod in range(self.resolution_log2 - 1))

        # Build TF expressions.
        with tf.name_scope('Dataset'), tf.device('/cpu:0'):
            self._tf_minibatch_in = tf.placeholder(
                tf.int64, name='minibatch_in', shape=[])

            for tfr_file, tfr_shape, tfr_lod in zip(tfr_files, tfr_shapes,
                                                    tfr_lods):
                if tfr_lod < 0:
                    continue
                dset = tf.data.TFRecordDataset(
                    tfr_file, compression_type='', buffer_size=buffer_mb << 20)
                dset = dset.map(parse_tfrecord_tf,
                                num_parallel_calls=num_threads)

                bytes_per_item = np.prod(tfr_shape) * \
                    np.dtype(self.dtype).itemsize
                if shuffle_mb > 0:
                    dset = dset.shuffle(
                        ((shuffle_mb << 20) - 1) // bytes_per_item + 1)
                if repeat:
                    dset = dset.repeat()
                if prefetch_mb > 0:
                    dset = dset.prefetch(
                        ((prefetch_mb << 20) - 1) // bytes_per_item + 1)
                dset = dset.batch(self._tf_minibatch_in)
                self._tf_datasets[tfr_lod] = dset
            self._tf_iterator = tf.data.Iterator.from_structure(
                self._tf_datasets[0].output_types,
                self._tf_datasets[0].output_shapes)
            self._tf_init_ops = {lod: self._tf_iterator.make_initializer(
                dset) for lod, dset in self._tf_datasets.items()}

    # Use the given minibatch size
    # and level-of-detail for the data returned by get_minibatch_tf().
    def configure(self, minibatch_size, lod=0):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 and lod in self._tf_datasets
        if self._cur_minibatch != minibatch_size or self._cur_lod != lod:
            self._tf_init_ops[lod].run({self._tf_minibatch_in: minibatch_size})
            self._cur_minibatch = minibatch_size
            self._cur_lod = lod

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self):  # => images,
        return self._tf_iterator.get_next()

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size, lod=0):  # => images,
        self.configure(minibatch_size, lod)
        if self._tf_minibatch_np is None:
            self._tf_minibatch_np = self.get_minibatch_tf()
        minibatch = tf.get_default_session().run(self._tf_minibatch_np)
        return minibatch


# ----------------------------------------------------------------------------
# Base class for datasets that are generated on the fly.
# ----------------------------------------------------------------------------
# Helper func for constructing a dataset object using the given options.


def load_dataset(data_dir=None,
                 verbose=False, **kwargs):
    adjusted_kwargs = dict(kwargs)
    if 'tfrecord_dir' in adjusted_kwargs and data_dir is not None:
        adjusted_kwargs['tfrecord_dir'] = os.path.join(
            data_dir, adjusted_kwargs['tfrecord_dir'])
    if verbose:
        print('Streaming data using TFRecordDataset...')
    dataset = TFRecordDataset(**adjusted_kwargs)
    if verbose:
        print('Dataset shape =', np.int32(dataset.shape).tolist())
        print('Dynamic range =', dataset.dynamic_range)

    return dataset

# ----------------------------------------------------------------------------
