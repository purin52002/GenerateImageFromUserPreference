from pathlib import Path
import tensorflow as tf


def load_GD(load_dir_path: str, is_compile=False):
    load_dir_path = Path(load_dir_path)

    G_path = str(load_dir_path / 'Generator.h5')
    D_path = str(load_dir_path/'Discriminator.h5')

    G = tf.keras.models.load_model(G_path, compile=is_compile)
    D = tf.keras.models.load_model(D_path, compile=is_compile)
    return G, D


def save_GD(G, D, save_dir_path: str, overwrite=False):
    save_dir_path = Path(save_dir_path)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    G_path = str(save_dir_path/'Generator.h5')
    D_path = str(save_dir_path/'Discriminator.h5')

    tf.keras.models.save_model(G, G_path, overwrite=overwrite)
    tf.keras.models.save_model(D, D_path, overwrite=overwrite)
    print(f'Save model to {str(save_dir_path)}')


def load_GD_weights(G, D, load_dir_path: str, by_name=True):
    load_dir_path = Path(load_dir_path)

    G_path = str(load_dir_path / 'Generator.h5')
    D_path = str(load_dir_path/'Discriminator.h5')

    G.load_weights(G_path, by_name=by_name)
    D.load_weights(D_path, by_name=by_name)
    return G, D


def save_GD_weights(G, D, save_dir_path: str):
    try:
        save_dir_path = Path(save_dir_path)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        G_path = str(save_dir_path/'Generator.h5')
        D_path = str(save_dir_path/'Discriminator.h5')

        G.save_weights(G_path)
        D.save_weights(D_path)
        print(f'Save weights to {str(save_dir_path)}')
    except ValueError:
        print('Save model snapshot failed!')
