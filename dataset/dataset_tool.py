# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys

import argparse

import numpy as np
import tensorflow as tf
import PIL.Image

import tfutil
import dataset

from pathlib import Path

# ----------------------------------------------------------------------------


def error(msg):
    print('Error: ' + msg)
    exit(1)

# ----------------------------------------------------------------------------


class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True,
                 progress_interval=10):
        self.tfrecord_dir = tfrecord_dir
        self.tfr_prefix = os.path.join(
            self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_images = expected_images
        self.cur_images = 0
        self.shape = None
        self.resolution_log2 = None
        self.tfr_writers = []
        self.print_progress = print_progress
        self.progress_interval = progress_interval
        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert(os.path.isdir(self.tfrecord_dir))

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
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2**self.resolution_log2
            tfr_opt = tf.python_io.TFRecordOptions(
                tf.python_io.TFRecordCompressionType.NONE)
            for lod in range(self.resolution_log2 - 1):
                tfr_file = self.tfr_prefix + \
                    '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(
                    tf.python_io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
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

# ----------------------------------------------------------------------------


def display(tfrecord_dir):
    print('Loading dataset "%s"' % tfrecord_dir)
    tfutil.init_tf({'gpu_options.allow_growth': True})
    dset = dataset.TFRecordDataset(
        tfrecord_dir, max_label_size='full', repeat=False, shuffle_mb=0)
    tfutil.init_uninited_vars()

    idx = 0
    while True:
        try:
            images, labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break
        if idx == 0:
            print('Displaying images')
            import cv2  # pip install opencv-python
            cv2.namedWindow('dataset_tool')
            print('Press SPACE or ENTER to advance, ESC to exit')
        print('\nidx = %-8d\nlabel = %s' % (idx, labels[0].tolist()))
        cv2.imshow('dataset_tool', images[0].transpose(
            1, 2, 0)[:, :, ::-1])  # CHW => HWC, RGB => BGR
        idx += 1
        if cv2.waitKey() == 27:
            break
    print('\nDisplayed %d images.' % idx)

# ----------------------------------------------------------------------------


def extract(tfrecord_dir, output_dir):
    print('Loading dataset "%s"' % tfrecord_dir)
    tfutil.init_tf({'gpu_options.allow_growth': True})
    dset = dataset.TFRecordDataset(
        tfrecord_dir, max_label_size=0, repeat=False, shuffle_mb=0)
    tfutil.init_uninited_vars()

    print('Extracting images to "%s"' % output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    idx = 0
    while True:
        if idx % 10 == 0:
            print('%d\r' % idx, end='', flush=True)
        try:
            images, labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break
        if images.shape[1] == 1:
            img = PIL.Image.fromarray(images[0][0], 'L')
        else:
            img = PIL.Image.fromarray(images[0].transpose(1, 2, 0), 'RGB')
        img.save(os.path.join(output_dir, 'img%08d.png' % idx))
        idx += 1
    print('Extracted %d images.' % idx)

# ----------------------------------------------------------------------------


def compare(tfrecord_dir_a, tfrecord_dir_b, ignore_labels):
    max_label_size = 0 if ignore_labels else 'full'
    print('Loading dataset "%s"' % tfrecord_dir_a)
    tfutil.init_tf({'gpu_options.allow_growth': True})
    dset_a = dataset.TFRecordDataset(
        tfrecord_dir_a, max_label_size=max_label_size, repeat=False,
        shuffle_mb=0)
    print('Loading dataset "%s"' % tfrecord_dir_b)
    dset_b = dataset.TFRecordDataset(
        tfrecord_dir_b, max_label_size=max_label_size, repeat=False, shuffle_mb=0)
    tfutil.init_uninited_vars()

    print('Comparing datasets')
    idx = 0
    identical_images = 0
    identical_labels = 0
    while True:
        if idx % 100 == 0:
            print('%d\r' % idx, end='', flush=True)
        try:
            images_a, labels_a = dset_a.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            images_a, labels_a = None, None
        try:
            images_b, labels_b = dset_b.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            images_b, labels_b = None, None
        if images_a is None or images_b is None:
            if images_a is not None or images_b is not None:
                print('Datasets contain different number of images')
            break
        if images_a.shape == images_b.shape and np.all(images_a == images_b):
            identical_images += 1
        else:
            print('Image %d is different' % idx)
        if labels_a.shape == labels_b.shape and np.all(labels_a == labels_b):
            identical_labels += 1
        else:
            print('Label %d is different' % idx)
        idx += 1
    print('Identical images: %d / %d' % (identical_images, idx))
    if not ignore_labels:
        print('Identical labels: %d / %d' % (identical_labels, idx))

# ----------------------------------------------------------------------------


def create_mnist(tfrecord_dir, mnist_dir):
    print('Loading MNIST from "%s"' % mnist_dir)
    import gzip
    with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    with gzip.open(os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as file:
        labels = np.frombuffer(file.read(), np.uint8, offset=8)
    images = images.reshape(-1, 1, 28, 28)
    images = np.pad(images, [(0, 0), (0, 0), (2, 2),
                             (2, 2)], 'constant', constant_values=0)
    assert images.shape == (60000, 1, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        tfr.add_labels(onehot[order])

# ----------------------------------------------------------------------------


def create_mnistrgb(tfrecord_dir, mnist_dir, num_images=1000000,
                    random_seed=123):
    print('Loading MNIST from "%s"' % mnist_dir)
    import gzip
    with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0, 0), (2, 2), (2, 2)],
                    'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255

    with TFRecordExporter(tfrecord_dir, num_images) as tfr:
        rnd = np.random.RandomState(random_seed)
        for idx in range(num_images):
            tfr.add_image(images[rnd.randint(images.shape[0], size=3)])

# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------


def execute_cmdline(argv):
    prog = argv[0]
    description = \
        'Tool for creating,'\
        'extracting,'\
        'and visualizing Progressive GAN datasets.'

    parser = argparse.ArgumentParser(description=description)

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (
            prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc,
                                     epilog=epilog)

    p = add_command('display',          'Display images in dataset.',
                    'display datasets/mnist')
    p.add_argument('tfrecord_dir',     help='Directory containing dataset')

    p = add_command('extract',          'Extract images from dataset.',
                    'extract datasets/mnist mnist-images')
    p.add_argument('tfrecord_dir',     help='Directory containing dataset')
    p.add_argument('output_dir',
                   help='Directory to extract the images into')

    p = add_command('compare',          'Compare two datasets.',
                    'compare datasets/mydataset datasets/mnist')
    p.add_argument('tfrecord_dir_a',
                   help='Directory containing first dataset')
    p.add_argument('tfrecord_dir_b',
                   help='Directory containing second dataset')
    p.add_argument('--ignore_labels',
                   help='Ignore labels (default: 0)', type=int, default=0)

    p = add_command('create_mnist',     'Create dataset for MNIST.',
                    'create_mnist datasets/mnist ~/downloads/mnist')
    p.add_argument('tfrecord_dir',
                   help='New dataset directory to be created')
    p.add_argument('mnist_dir',        help='Directory containing MNIST')

    p = add_command('create_mnistrgb',  'Create dataset for MNIST-RGB.',
                    'create_mnistrgb datasets/mnistrgb ~/downloads/mnist')
    p.add_argument('tfrecord_dir',
                   help='New dataset directory to be created')
    p.add_argument('mnist_dir',        help='Directory containing MNIST')
    p.add_argument('--num_images',
                   help='Number of composite images to create (default: 1000000)', type=int, default=1000000)
    p.add_argument('--random_seed',
                   help='Random seed (default: 123)', type=int, default=123)

    p = add_command('create_cifar10',   'Create dataset for CIFAR-10.',
                    'create_cifar10 datasets/cifar10 ~/downloads/cifar10')
    p.add_argument('tfrecord_dir',
                   help='New dataset directory to be created')
    p.add_argument('cifar10_dir',      help='Directory containing CIFAR-10')

    p = add_command('create_cifar100',  'Create dataset for CIFAR-100.',
                    'create_cifar100 datasets/cifar100 ~/downloads/cifar100')
    p.add_argument('tfrecord_dir',
                   help='New dataset directory to be created')
    p.add_argument('cifar100_dir',     help='Directory containing CIFAR-100')

    p = add_command('create_svhn',      'Create dataset for SVHN.',
                    'create_svhn datasets/svhn ~/downloads/svhn')
    p.add_argument('tfrecord_dir',
                   help='New dataset directory to be created')
    p.add_argument('svhn_dir',         help='Directory containing SVHN')

    p = add_command('create_lsun',      'Create dataset for single LSUN category.',
                    'create_lsun datasets/lsun-car-100k ~/downloads/lsun/car_lmdb --resolution 256 --max_images 100000')
    p.add_argument('tfrecord_dir',
                   help='New dataset directory to be created')
    p.add_argument(
        'lmdb_dir',         help='Directory containing LMDB database')
    p.add_argument('--resolution',
                   help='Output resolution (default: 256)', type=int, default=256)
    p.add_argument('--max_images',
                   help='Maximum number of images (default: none)', type=int, default=None)

    p = add_command('create_celeba',    'Create dataset for CelebA.',
                    'create_celeba datasets/celeba ~/downloads/celeba')
    p.add_argument('tfrecord_dir',
                   help='New dataset directory to be created')
    p.add_argument('celeba_dir',       help='Directory containing CelebA')
    p.add_argument(
        '--cx',             help='Center X coordinate (default: 89)', type=int, default=89)
    p.add_argument(
        '--cy',             help='Center Y coordinate (default: 121)', type=int, default=121)

    p = add_command('create_celebahq',  'Create dataset for CelebA-HQ.',
                    'create_celebahq datasets/celebahq ~/downloads/celeba ~/downloads/celeba-hq-deltas')
    p.add_argument('tfrecord_dir',
                   help='New dataset directory to be created')
    p.add_argument('celeba_dir',       help='Directory containing CelebA')
    p.add_argument('delta_dir',
                   help='Directory containing CelebA-HQ deltas')
    p.add_argument('--num_threads',
                   help='Number of concurrent threads (default: 4)', type=int, default=4)
    p.add_argument('--num_tasks',
                   help='Number of concurrent processing tasks (default: 100)', type=int, default=100)

    p = add_command('create_from_images', 'Create dataset from a directory full of images.',
                    'create_from_images datasets/mydataset myimagedir')
    p.add_argument('tfrecord_dir',
                   help='New dataset directory to be created')
    p.add_argument('image_dir',        help='Directory containing the images')
    p.add_argument('--shuffle',
                   help='Randomize image order (default: 1)', type=int, default=1)

    p = add_command('create_from_hdf5', 'Create dataset from legacy HDF5 archive.',
                    'create_from_hdf5 datasets/celebahq ~/downloads/celeba-hq-1024x1024.h5')
    p.add_argument('tfrecord_dir',
                   help='New dataset directory to be created')
    p.add_argument('hdf5_filename',
                   help='HDF5 archive containing the images')
    p.add_argument('--shuffle',
                   help='Randomize image order (default: 1)', type=int, default=1)

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    execute_cmdline(sys.argv)

# ----------------------------------------------------------------------------
