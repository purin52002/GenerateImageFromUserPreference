from ImageEnhancer.image_enhancer import ImageEnhancer
from ScoredParamIO.scored_param_reader import format_scored_param_file_list
from TrainDataGenerator.TFRecordsMaker.util import DATASET_TYPE_LIST
from conpare import convert as compare_convert
from regression import convert as regression_convert
from UserPreferencePredictor.Model.util \
    import MODEL_TYPE_LIST
from pathlib import Path
from argparse import ArgumentParser


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-d', '--dataset_dir_path', required=True)
    parser.add_argument('-p', '--param_paths', nargs='*', required=True)
    parser.add_argument('-i', '--image_path')
    parser.add_argument('-t', '--model_type', choices=MODEL_TYPE_LIST,
                        required=True)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


if __name__ == "__main__":
    args = _get_args()
    image_enhancer = ImageEnhancer(args.image_path)
    scored_param_list = format_scored_param_file_list(args.param_paths)

    if not Path(args.dataset_dir_path).is_dir():
        raise NotADirectoryError

    rate_dict = \
        dict(zip(DATASET_TYPE_LIST, [0.7, 0.2, 0.1]))

    convert_func_dict = \
        dict(zip(MODEL_TYPE_LIST, [compare_convert, regression_convert]))

    convert_func_dict[args.model_type](args.dataset_dir_path, image_enhancer,
                                       scored_param_list, rate_dict)
