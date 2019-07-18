from tkinter import filedialog, messagebox
from TrainDataGenerator.TFRecordsMaker.util \
    import TRAINDATA_PATH

from UserPreferencePredictor.Model.Compare.ranknet import RankNet

from UserPreferencePredictor.Model.Regression.trainable_model \
    import TrainableModel as regression_trainable
from UserPreferencePredictor.Model.Regression.predictable_model \
    import PredictableModel as regression_predictable

from argparse import ArgumentParser


def get_load_dir(model_type=None):
    base_model_name = 'モデル'
    model_type_name = ''

    if model_type is COMPARE:
        model_type_name = '比較'

    if model_type is REGRESSION:
        model_type_name = '回帰'

    dialog_title = '学習済みの%sをロードするフォルダを選択してください' \
        % (model_type_name+base_model_name)

    load_dir = \
        filedialog.askdirectory(title=dialog_title, initialdir=TRAINDATA_PATH)

    if not load_dir:
        raise FileNotFoundError('フォルダが選択されませんでした')

    return load_dir


COMPARE, REGRESSION = 'compare', 'regression'
MODEL_TYPE_LIST = (COMPARE, REGRESSION)

TRAINABLE, PREDICTABLE = 'trainable', 'predictable'
USE_TYPE_LIST = (TRAINABLE, PREDICTABLE)

COMPARE_MODEL_BUILDER_DICT = \
    dict(zip(USE_TYPE_LIST,
             [RankNet]))

REGRESSION_BUILDER_DICT = \
    dict(zip(USE_TYPE_LIST,
             [regression_trainable,  regression_predictable]))

MODEL_BUILDER_DICT = \
    dict(zip(MODEL_TYPE_LIST,
             [COMPARE_MODEL_BUILDER_DICT, REGRESSION_BUILDER_DICT]))


def select_model_type():
    is_use_compare = messagebox.askyesno('modelの選択', 'はいなら比較、いいえなら回帰')

    model_type = COMPARE if is_use_compare else REGRESSION

    return model_type


def set_model_type_args(parser: ArgumentParser):
    parser.add_argument('model_type', choices=MODEL_TYPE_LIST)

    return parser
