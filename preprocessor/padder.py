import cv2


def mirror_padding(image):
    padding_y = image.shape[0] // 10
    padding_x = image.shape[1] // 10
    padded_image = cv2.copyMakeBorder(image, padding_y, padding_y,
                                      padding_x, padding_x,
                                      cv2.BORDER_REFLECT_101)

    return padded_image


def _get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-i', '--image_dir_path', required=True)
    parser.add_argument('-o', '--extract_dir_path', required=True)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


if __name__ == "__main__":
    from pathlib import Path
    args = _get_args()

    source_path = Path(args.image_dir_path)

    if not source_path.exists():
        raise FileNotFoundError

    if not source_path.is_dir():
        raise NotADirectoryError

    dest_path = Path(args.extract_dir_path)
    dest_path.mkdir(parents=True, exists_ok=True)

    for image_path in source_path.glob('*'):
        image = cv2.imread(str(image_path))

        padded_image = mirror_padding(image)
        print('padding %s' % image_path.name)

        cv2.imwrite(str(dest_path/image_path.name), padded_image)
