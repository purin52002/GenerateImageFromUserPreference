from TrainDataGenerator.TFRecordsMaker.util \
    import make_dataset_path_dict, TRAIN, VALIDATION
from UserPreferencePredictor.Model.Compare.ranknet import RankNet
from UserPreferencePredictor.Model.Compare.dataset import make_dataset
from UserPreferencePredictor.Model.util import MODEL_TYPE_LIST, COMPARE
from argparse import ArgumentParser

from datetime import datetime
from pathlib import Path


def _make_summary_dir(summary_dir_path: str):
    now = datetime.now()
    path = Path(summary_dir_path)/'{0:%m%d}'.format(now)/'{0:%H%M}'.format(now)

    if path.exists():
        path = Path(str(path.parent)+'_{0:%S}'.format(now))

    path.mkdir(parents=True)

    return str(path)


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-d', '--dataset_dir_path', required=True)
    parser.add_argument('-s', '--summary_dir_path', required=True)
    parser.add_argument('-l', '--load_dir_path')
    parser.add_argument('-t', '--model_type', choices=MODEL_TYPE_LIST,
                        required=True)
    parser.add_argument('-j', '--use_jupyter', action='store_true')

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


if __name__ == "__main__":
    args = _get_args()

    model_type = args.model_type

    batch_size = 100 if model_type == COMPARE else 10
    log_dir_path = _make_summary_dir(args.summary_dir_path)

    trainable_model = RankNet()

    if args.load_dir_path:
        try:
            trainable_model.load(args.load_dir_path)
        except ValueError:
            pass

    dataset_path_dict = make_dataset_path_dict(args.dataset_dir_path)
    dataset = {key: make_dataset(dataset_path_dict[key], batch_size, key)
               for key in [TRAIN, VALIDATION]}

    trainable_model.train(dataset[TRAIN], log_dir_path=log_dir_path,
                          valid_dataset=dataset[VALIDATION], epochs=30)

    trainable_model.save(log_dir_path)
