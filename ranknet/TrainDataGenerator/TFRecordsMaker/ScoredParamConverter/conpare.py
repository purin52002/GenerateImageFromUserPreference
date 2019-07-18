from tqdm import tqdm
from tkinter import Tk
import numpy as np

from ImageEnhancer.util import get_image_enhancer, ImageEnhancer
from ScoredParamIO.scored_param_reader import get_scored_param_list

from TrainDataGenerator.TFRecordsMaker.util \
    import get_dataset_save_dir


from TrainDataGenerator.TFRecordsMaker.switchable_writer \
    import AutoSwitchableWriter
from TrainDataGenerator.TFRecordsMaker.compare_maker import CompareMaker


def _make_label(left_score: float, right_score: float):
    if left_score > right_score:
        return 0

    elif right_score > left_score:
        return 1

    else:
        raise ValueError('score is same')


def convert(save_file_dir: str, image_enhancer: ImageEnhancer,
            scored_param_list: list, rate_dict: dict):
    scored_param_length = len(scored_param_list)

    data_length = 0

    for left_index in range(0, scored_param_length-1):
        for right_index in range(left_index+1, scored_param_length):
            left_param = scored_param_list[left_index]
            right_param = scored_param_list[right_index]

            left_score = scored_param_list[left_index]['score']
            right_score = scored_param_list[right_index]['score']

            if left_score != right_score:
                data_length += 1

    writer = AutoSwitchableWriter(save_file_dir, rate_dict, data_length)
    compare_maker = CompareMaker(writer)

    progress = tqdm(total=data_length)
    for left_index in range(0, scored_param_length-1):
        for right_index in range(left_index+1, scored_param_length):
            left_param = scored_param_list[left_index]
            right_param = scored_param_list[right_index]

            left_image = image_enhancer.resized_enhance(left_param)
            right_image = image_enhancer.resized_enhance(right_param)

            left_array = np.asarray(left_image)
            right_array = np.asarray(right_image)

            left_score = left_param['score']
            right_score = right_param['score']

            try:
                label = _make_label(left_score, right_score)
                compare_maker.write(left_array, right_array, label)
                progress.update()
            except ValueError:
                pass


if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    root.attributes('-topmost', True)
    root.lift()
    root.focus_force()

    image_enhancer = get_image_enhancer()
    scored_param_list = get_scored_param_list()
    save_file_dir = get_dataset_save_dir()
    root.destroy()

    rate_dict = \
        dict(zip(AutoSwitchableWriter.DATASET_TYPE_LIST, [0.7, 0.2, 0.1]))

    convert(save_file_dir, image_enhancer, scored_param_list, rate_dict)

    print('\n')

    print('--- complete ! ---')
