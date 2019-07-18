import cv2

SIZE = 128, 128


def resize_image(image, size: tuple):
    return cv2.resize(image, SIZE)


def _get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-i', '--image_dir_path', required=True)
    parser.add_argument('-o', '--resized_dir_path', required=True)
    parser.add_argument('-s', '--size', nargs=2, type=int, default=SIZE)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


if __name__ == "__main__":
    from pathlib import Path

    args = _get_args()

    source_path = Path(args.image_dir_path)

    if not source_path.exist():
        raise FileNotFoundError

    if not source_path.is_dir():
        raise NotADirectoryError

    dest_path = Path(args.resized_dir_path)
    dest_path.mkdir(parents=True, exists_ok=True)

    for image_path in source_path.glob('*'):
        image = cv2.imread(str(image_path))
        resize = resize_image(image)
        print('resize: %s' % str(image_path.name))
        cv2.imwrite(str(dest_path/image_path.name), resize)
