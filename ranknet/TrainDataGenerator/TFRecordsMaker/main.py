from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import tensorflow as tf
import json
from record import get_record_yield
from combination import get_combination_yield, get_data_length


def _convert(save_file_path: str, scored_dir_path_list: list):
    data_length = get_data_length(scored_dir_path_list)

    writer = tf.io.TFRecordWriter(save_file_path)

    progress = tqdm(total=data_length)
    for parsed_info in get_combination_yield(scored_dir_path_list):
        for record in get_record_yield(parsed_info):
            writer.write(record)
            progress.update()


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-i', '--score_file_path', required=True)
    parser.add_argument('-o', '--dataset_dir_path', required=True)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


if __name__ == "__main__":
    args = _get_args()

    if not Path(args.score_file_path).exists():
        raise FileNotFoundError

    if not Path(args.score_file_path).suffix == '.json':
        raise ValueError

    dataset_dir_path = Path(args.dataset_dir_path)
    dataset_dir_path.mkdir(parents=True, exist_ok=True)

    train_records_path = str(dataset_dir_path/'train.tfrecords')

    scored_dir_path_list = list()
    with open(args.score_file_path) as fp:
        scored_dir_path_list = json.load(fp)

    _convert(train_records_path, scored_dir_path_list)

    print('--- complete ! ---')
