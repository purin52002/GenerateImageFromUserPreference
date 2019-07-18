import cv2


def _get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('-i', '--image_dir_path', required=True)
    parser.add_argument('-o', '--extract_dir_path', required=True)
    parser.add_argument('-t', '--assorted_file_path', required=True)

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

    with open(args.assorted_file_path) as f:
        if not f:
            exit()

        for image_name_list in f:
            for image_name in image_name_list.split(','):
                image = cv2.imread(str(source_path/image_name))
                if image is None:
                    print(f'{image_name} is not found')
                    exit()

                cv2.imwrite(str(dest_path/image_name), image)
                print('assort: %s' % image_name)
