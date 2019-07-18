from tkinter import Tk

from ImageEnhancer.util import get_image_enhancer
from ImageEnhancer.enhance_definer import enhance_name_list
from ScoredParamIO.scored_param_reader import get_scored_param_list

from UserPreferencePredictor.Model.util \
    import MODEL_BUILDER_DICT, ModelType, get_load_dir, UseType

from pprint import pprint

MODEL_KEY_LIST = ['compare', 'regression']


def _get_load_dir_dict():
    try:
        compare_load_dir = get_load_dir(ModelType.COMPARE)
    except FileNotFoundError:
        print('比較モデルのロード用フォルダが選択されなかったため終了します')
        exit()

    try:
        regression_load_dir = get_load_dir(ModelType.REGRESSION)
    except FileNotFoundError:
        print('回帰モデルのロード用フォルダが選択されなかったため終了します')
        exit()

    return dict(zip(MODEL_KEY_LIST, [compare_load_dir, regression_load_dir]))


def _calc_param_error(score_base: dict, predict_base: dict):
    return sum([abs(score_base['param'][param_key] -
                    predict_base['param'][param_key])
                for param_key in enhance_name_list])/len(enhance_name_list)


def _calc_error_data(score_base_list: list, predict_base_list: list):
    assert len(score_base_list) == len(predict_base_list)

    return sum([_calc_param_error(score_base, predict_base)
                for score_base, predict_base
                in zip(score_base_list, predict_base_list)]) / \
        len(score_base_list)


if __name__ == "__main__":
    root = Tk()
    root.attributes('-topmost', True)

    root.withdraw()
    root.lift()
    root.focus_force()

    load_dir_dict = _get_load_dir_dict()

    predict_model_dict = \
        {
            key: MODEL_BUILDER_DICT[model_type][UseType.PREDICTABLE]()
            for key, model_type in zip(MODEL_KEY_LIST, list(ModelType))
        }

    for key, value in predict_model_dict.items():
        try:
            value.restore(load_dir_dict[key])
        except ValueError:
            print('%s model is not restore' % key)
            exit()

    image_enhancer = get_image_enhancer()
    try:
        scored_param_list = get_scored_param_list()
    except FileNotFoundError:
        exit()

    root.destroy()

    data_list = [
        {
            'param': scored_param,
            'image': image_enhancer.enhance(scored_param),
            'score': scored_param['score']}
        for scored_param in scored_param_list
    ]

    evaluate_list_dict = \
        {model_key: predict_model.predict_evaluate(data_list).tolist()
         for model_key, predict_model in predict_model_dict.items()}

    for i in range(len(data_list)):
        for model_key in MODEL_KEY_LIST:
            data_list[i]['%s_evaluate' % model_key] = \
                evaluate_list_dict[model_key][i][0]

    sorted_scored_data_list = \
        sorted(data_list, key=lambda x: x['score'], reverse=True)

    sorted_predicted_data_dict = \
        {model_key:
         sorted(
             data_list, key=lambda x: x['%s_evaluate' % model_key],
             reverse=True)
         for model_key in MODEL_KEY_LIST}

    error_dict = \
        {model_key:
         _calc_error_data(sorted_scored_data_list,
                          sorted_predicted_data_dict[model_key])
         for model_key in MODEL_KEY_LIST}

    pprint(error_dict)
