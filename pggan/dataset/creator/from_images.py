import numpy as np
from exception import DatasetCreatorException
from pathlib import Path
from PIL import Image
from exporter import TFRecordExporter


class FromImageException(DatasetCreatorException):
    pass


def create_from_images(tfrecord_dir_path: str, image_dir_path: str,
                       is_shuffle: bool):
    image_dir_path = Path(image_dir_path)
    print(f'Loading images from {str(image_dir_path)}')

    image_filenames = sorted(list(image_dir_path.glob('*')))

    if len(image_filenames) == 0:
        raise FromImageException('No input images found')

    img = np.asarray(Image.open(image_filenames[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1

    if img.shape[1] != resolution:
        raise FromImageException(
            'Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        raise FromImageException(
            'Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        raise FromImageException(
            'Input images must be stored as RGB or grayscale')

    with TFRecordExporter(tfrecord_dir_path, len(image_filenames)) as tfr:
        order = \
            tfr.choose_shuffled_order() \
            if is_shuffle else \
            np.arange(len(image_filenames))

        for idx in range(order.size):
            img = np.asarray(Image.open(image_filenames[order[idx]]))
            if channels == 1:
                img = img[np.newaxis, :, :]  # HW => CHW
            else:
                img = img.transpose(2, 0, 1)  # HWC => CHW
            tfr.add_image(img)


COMMAND = 'from_image'


def add_command_line(subparsers):
    description = 'Create dataset from a directory full of images.'
    parser = subparsers.add_parser(COMMAND, description=description)

    parser.add_argument('-out', '--tfrecord_dir_path', required=True,
                        help='New dataset directory to be created')

    parser.add_argument('-in', '--image_dir_path', required=True,
                        help='Directory containing the images')

    parser.add_argument('-s', '--is_shuffle', help='Randomize image order',
                        type=bool, default=1)


func_dict = {COMMAND: create_from_images}
