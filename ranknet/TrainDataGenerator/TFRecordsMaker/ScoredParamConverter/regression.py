from tqdm import tqdm
from tkinter import Tk
import numpy as np

from TrainDataGenerator.TFRecordsMaker.util \
    import get_dataset_save_dir

from ImageEnhancer.util import get_image_enhancer, ImageEnhancer
from ScoredParamIO.scored_param_reader import get_scored_param_list

from TrainDataGenerator.TFRecordsMaker.switchable_writer \
    import AutoSwitchableWriter
from TrainDataGenerator.TFRecordsMaker.regression_maker import RegressionMaker


def convert(save_file_dir: str, image_enhancer: ImageEnhancer,
            scored_param_list: list, rate_dict: dict):

    data_length = len(scored_param_list)

    writer = \
        AutoSwitchableWriter(save_file_dir, rate_dict, data_length)
    regression_maker = RegressionMaker(writer)

    max_score = max(scored_param_list, key=lambda x: x['score'])['score']

    for scored_param \
            in tqdm(scored_param_list, desc='write TFRecords'):

        image = image_enhancer.resized_enhance(scored_param)
        image_array = np.asarray(image)

        score = scored_param['score']/max_score

        regression_maker.write(image_array, score)


if __name__ == "__main__":
    root = Tk()
    root.attributes('-topmost', True)

    root.withdraw()
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
