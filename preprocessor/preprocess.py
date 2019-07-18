from extractor import extract_face
from resizer import resize_image, SIZE
from padder import mirror_padding

import cv2
from pathlib import Path
from argparse import ArgumentParser


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-i', '--image_dir_path', required=True)
    parser.add_argument('-o', '--processed_dir_path', required=True)
    parser.add_argument('-s', '--size', nargs=2, type=int, default=SIZE)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


if __name__ == "__main__":
    args = _get_args()

    source_path = Path(args.image_dir_path)

    if not source_path.exists():
        raise FileNotFoundError

    if not source_path.is_dir():
        raise NotADirectoryError

    dest_path = Path(args.processed_dir_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    for image_path in source_path.glob('*'):
        image = cv2.imread(str(image_path))
        extract_list = extract_face(image)
        print(f'extract: {image_path.name}')

        for i, extract in enumerate(extract_list, 1):
            if extract.size == 0:
                continue

            processed_image = mirror_padding(resize_image(extract, args.size))
            cv2.imwrite(str(dest_path/f'{i}.{image_path.name}'),
                        processed_image)

    print('complete')