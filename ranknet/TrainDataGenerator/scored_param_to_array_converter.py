from ImageEnhancer.image_enhancer import ImageEnhancer
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from ScoredParamIO.scored_param_reader import read_scored_param
import numpy as np
from pathlib import Path

IMAGE_WIDTH, IMAGE_HEIGHT = 32, 32


def _file_select_dialog():
    image_path = filedialog.askopenfilename(title='画像を選択してください', filetypes=[
                                            ('image file', ['.png', '.jpg'])])
    if not image_path:
        exit()

    image_enhancer = ImageEnhancer(image_path)

    param_file_list = filedialog.askopenfiles(
        'r', title='スコアリングされたパラメータデータを選択してください',
        filetypes=[('scored param file', ['.csv'])])
    if not param_file_list:
        exit()

    scored_param_list = []
    for param_file in param_file_list:
        scored_param_list.extend(read_scored_param(param_file))

    save_dir = filedialog.askdirectory(title='arrayデータを保存するフォルダを選択してください')
    if not save_dir:
        exit()

    return image_enhancer, scored_param_list, save_dir


def _disp_npz(data_dir: str):
    npz = np.load(str(Path(data_dir).joinpath('image_00001.npz')))
    for key in npz.files:
        print(key)
        print(npz[key])


if __name__ == "__main__":
    root = tk.Tk()
    root.attributes('-topmost', True)

    root.withdraw()
    root.lift()
    root.focus_force()
    image_enhancer, scored_param_list, save_dir = _file_select_dialog()

    root.destroy()

    for index, scored_param in tqdm(enumerate(scored_param_list),
                                    desc='convert image to array'):
        resized_image = image_enhancer.enhance(scored_param) \
            .resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        score = scored_param['score']

        np.savez(str(Path(save_dir).joinpath('image_%05d' % (index))),
                 image_array=np.asarray(resized_image).astype(np.float32)/255,
                 score=np.array([score]))

    print('--- complete ! ---')
