import time

import tensorflow as tf

import misc
from pathlib import Path
import math
import numpy as np

from model.filer import save_GD_weights, save_GD
from model.pg_gan import build_trainable_model

import config
from dataset.dataset import TFRecordDataset

K = tf.keras.backend
optimizers = tf.keras.optimizers


def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0


def rampdown_linear(epoch, num_epochs, rampdown_length):
    if epoch >= num_epochs - rampdown_length:
        return float(num_epochs - epoch) / rampdown_length
    else:
        return 1.0


def create_result_subdir(result_dir_path: str, run_desc):
    result_dir_path = Path(result_dir_path)
    # Select run ID and create subdir.
    while True:
        run_id = 0
        for fname in result_dir_path.glob('*'):
            try:
                fbase = fname.name
                ford = int(fbase[:fbase.find('-')])
                run_id = max(run_id, ford + 1)
            except ValueError:
                pass

        result_subdir_path = result_dir_path/f'{run_id:03}-{run_desc}'

        if not result_subdir_path.exists():
            result_subdir_path.mkdir(parents=True)
            break

    print(f'Saving results to {str(result_subdir_path)}')
    return result_subdir_path


def random_latents(num_latents, G_input_shape):
    return np.random.randn(num_latents, *G_input_shape[1:]).astype(np.float32)


def random_labels(num_labels, training_set):
    return training_set.labels[np.random.randint(training_set.labels.shape[0],
                                                 size=num_labels)]


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def multiple_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)


def mean_loss(y_true, y_pred):
    return K.mean(y_pred)


def load_dataset(dataset_spec=None, verbose=True, **spec_overrides):
    if verbose:
        print('Loading dataset...')

    if dataset_spec is None:
        dataset_spec = config.dataset

    data_dir_path = Path(config.data_dir)

    # take a copy of the dict before modifying it
    dataset_spec = dict(dataset_spec)
    dataset_spec.update(spec_overrides)
    dataset_spec['h5_path'] = str(data_dir_path / dataset_spec['h5_path'])

    if 'label_path' in dataset_spec:
        dataset_spec['label_path'] = \
            str(data_dir_path / dataset_spec['label_path'])
    training_set = TFRecordDataset(**config.dataset.dataset_arg_dict)

    if verbose:
        print('Dataset shape =', np.int32(training_set.shape).tolist())

    drange_orig = training_set.get_dynamic_range()
    if verbose:
        print('Dynamic range =', drange_orig)

    return training_set, drange_orig


speed_factor = 20


def train_gan(
    separate_funcs=False,
    D_training_repeats=1,
    G_learning_rate_max=0.0010,
    D_learning_rate_max=0.0010,
    G_smoothing=0.999,
    adam_beta1=0.0,
    adam_beta2=0.99,
    adam_epsilon=1e-8,
    minibatch_default=16,
    minibatch_overrides={},
    rampup_kimg=40/speed_factor,
    rampdown_kimg=0,
    lod_initial_resolution=4,
    lod_training_kimg=400/speed_factor,
    lod_transition_kimg=400/speed_factor,
    total_kimg=10000/speed_factor,
    dequantize_reals=False,
    gdrop_beta=0.9,
    gdrop_lim=0.5,
    gdrop_coef=0.2,
    gdrop_exp=2.0,
    drange_net=[-1, 1],
    drange_viz=[-1, 1],
    image_grid_size=None,
    tick_kimg_default=50/speed_factor,
    tick_kimg_overrides={32: 20, 64: 10, 128: 10, 256: 5, 512: 2, 1024: 1},
    image_snapshot_ticks=1,
    network_snapshot_ticks=4,
    image_grid_type='default',
    # resume_network          = '000-celeba/network-snapshot-000488',
    resume_network=None,
    resume_kimg=0.0,
        resume_time=0.0):

    training_set, drange_orig = load_dataset()

    G, G_train,\
        D, D_train = build_trainable_model(resume_network,
                                           training_set.shape[3],
                                           training_set.shape[1],
                                           training_set.labels.shape[1],
                                           adam_beta1, adam_beta2,
                                           adam_epsilon)

    # Misc init.
    resolution_log2 = int(np.round(np.log2(G.output_shape[2])))
    initial_lod = max(resolution_log2 -
                      int(np.round(np.log2(lod_initial_resolution))), 0)
    cur_lod = 0.0
    min_lod, max_lod = -1.0, -2.0
    fake_score_avg = 0.0

    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()

    train_start_time = tick_start_time - resume_time

    if image_grid_type == 'default':
        if image_grid_size is None:
            w, h = G.output_shape[1], G.output_shape[2]
            print('w:%d,h:%d' % (w, h))
            image_grid_size = (np.clip(int(1920 // w), 3, 16).astype('int'),
                               np.clip(1080 / h, 2, 16).astype('int'))

        print('image_grid_size:', image_grid_size)

        example_real_images, snapshot_fake_labels = \
            training_set.get_random_minibatch_channel_last(
                np.prod(image_grid_size), labels=True)

        snapshot_fake_latents = random_latents(
            np.prod(image_grid_size), G.input_shape)
    else:
        raise ValueError('Invalid image_grid_type', image_grid_type)

    result_subdir = misc.create_result_subdir(
        config.result_dir, config.run_desc)
    result_subdir = Path(result_subdir)

    print('example_real_images.shape:', example_real_images.shape)
    misc.save_image_grid(example_real_images,
                         str(result_subdir / 'reals.png'),
                         drange=drange_orig, grid_size=image_grid_size)

    snapshot_fake_latents = random_latents(
        np.prod(image_grid_size), G.input_shape)
    snapshot_fake_images = G.predict_on_batch(snapshot_fake_latents)
    misc.save_image_grid(snapshot_fake_images,
                         str(result_subdir/f'fakes{cur_nimg//1000:06}.png'),
                         drange=drange_viz, grid_size=image_grid_size)

    # nimg_h = 0

    while cur_nimg < total_kimg * 1000:

        # Calculate current LOD.
        cur_lod = initial_lod
        if lod_training_kimg or lod_transition_kimg:
            tlod = (cur_nimg / (1000.0/speed_factor)) / \
                (lod_training_kimg + lod_transition_kimg)
            cur_lod -= np.floor(tlod)
            if lod_transition_kimg:
                cur_lod -= max(1.0 + (np.fmod(tlod, 1.0) - 1.0) *
                               (lod_training_kimg + lod_transition_kimg) /
                               lod_transition_kimg,
                               0.0)
            cur_lod = max(cur_lod, 0.0)

        # Look up resolution-dependent parameters.
        cur_res = 2 ** (resolution_log2 - int(np.floor(cur_lod)))
        minibatch_size = minibatch_overrides.get(cur_res, minibatch_default)
        tick_duration_kimg = tick_kimg_overrides.get(
            cur_res, tick_kimg_default)

        # Update network config.
        lrate_coef = rampup(cur_nimg / 1000.0, rampup_kimg)
        lrate_coef *= rampdown_linear(cur_nimg /
                                      1000.0, total_kimg, rampdown_kimg)

        K.set_value(G.optimizer.lr, np.float32(
            lrate_coef * G_learning_rate_max))
        K.set_value(G_train.optimizer.lr, np.float32(
            lrate_coef * G_learning_rate_max))

        K.set_value(D_train.optimizer.lr, np.float32(
            lrate_coef * D_learning_rate_max))
        if hasattr(G_train, 'cur_lod'):
            K.set_value(G_train.cur_lod, np.float32(cur_lod))
        if hasattr(D_train, 'cur_lod'):
            K.set_value(D_train.cur_lod, np.float32(cur_lod))

        new_min_lod, new_max_lod = int(
            np.floor(cur_lod)), int(np.ceil(cur_lod))
        if min_lod != new_min_lod or max_lod != new_max_lod:
            min_lod, max_lod = new_min_lod, new_max_lod

        # train D
        d_loss = None
        for idx in range(D_training_repeats):
            mb_reals, mb_labels = \
                training_set.get_random_minibatch_channel_last(
                    minibatch_size, lod=cur_lod, shrink_based_on_lod=True,
                    labels=True)
            mb_latents = random_latents(minibatch_size, G.input_shape)

            # compensate for shrink_based_on_lod
            if min_lod > 0:
                mb_reals = np.repeat(mb_reals, 2**min_lod, axis=1)
                mb_reals = np.repeat(mb_reals, 2**min_lod, axis=2)

            mb_fakes = G.predict_on_batch([mb_latents])

            epsilon = np.random.uniform(0, 1, size=(minibatch_size, 1, 1, 1))
            interpolation = epsilon*mb_reals + (1-epsilon)*mb_fakes
            mb_reals = misc.adjust_dynamic_range(
                mb_reals, drange_orig, drange_net)
            d_loss, d_diff, d_norm = \
                D_train.train_on_batch([mb_fakes, mb_reals, interpolation],
                                       [np.ones((minibatch_size, 1, 1, 1)),
                                        np.ones((minibatch_size, 1))])
            d_score_real = D.predict_on_batch(mb_reals)
            d_score_fake = D.predict_on_batch(mb_fakes)
            print('real score: %d fake score: %d' %
                  (np.mean(d_score_real), np.mean(d_score_fake)))
            cur_nimg += minibatch_size

        # train G
        mb_latents = random_latents(minibatch_size, G.input_shape)

        g_loss = G_train.train_on_batch(
            [mb_latents], (-1)*np.ones((mb_latents.shape[0], 1, 1, 1)))

        print('%d [D loss: %f] [G loss: %f]' % (cur_nimg, d_loss, g_loss))

        fake_score_cur = np.clip(np.mean(d_loss), 0.0, 1.0)
        fake_score_avg = fake_score_avg * gdrop_beta + \
            fake_score_cur * (1.0 - gdrop_beta)
        gdrop_strength = gdrop_coef * \
            (max(fake_score_avg - gdrop_lim, 0.0) ** gdrop_exp)
        if hasattr(D, 'gdrop_strength'):
            K.set_value(D.gdrop_strength, np.float32(gdrop_strength))

        is_complete = cur_nimg >= total_kimg * 1000
        is_generate_a_lot = \
            cur_nimg >= tick_start_nimg + tick_duration_kimg * 1000

        if is_generate_a_lot or is_complete:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time

            print(f'tick time: {tick_time}')
            print(f'tick image: {tick_kimg}k')
            tick_start_time = cur_time

            # Visualize generated images.
            is_image_snapshot_ticks = cur_tick % image_snapshot_ticks == 0
            if is_image_snapshot_ticks or is_complete:
                snapshot_fake_images = G.predict_on_batch(
                    snapshot_fake_latents)
                misc.save_image_grid(snapshot_fake_images,
                                     str(result_subdir /
                                         f'fakes{cur_nimg // 1000:06}.png'),
                                     drange=drange_viz,
                                     grid_size=image_grid_size)

            if cur_tick % network_snapshot_ticks == 0 or is_complete:
                save_GD_weights(G, D, str(
                    result_subdir / f'network-snapshot-{cur_nimg // 1000:06}'))
        break
    save_GD(G, D, str(result_subdir/'network-final'))
    training_set.close()
    print('Done.')

    train_complete_time = time.time()-train_start_time
    print(f'training time: {train_complete_time}')


if __name__ == '__main__':
    np.random.seed(config.random_seed)

    train_gan(**config.train)
