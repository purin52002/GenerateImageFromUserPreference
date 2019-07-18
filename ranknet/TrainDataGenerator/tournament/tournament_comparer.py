from comparer import CompareCanvasGroupFrame

from tkinter import Tk
from ImageEnhancer.util import ImageEnhancer
from ImageEnhancer.enhance_definer import enhance_name_list

from argparse import ArgumentParser
from TrainDataGenerator.image_parameter_generator \
    import generate_image_parameter_list
from game import TournamentGame
from logging import StreamHandler, DEBUG

from pathlib import Path


def _get_args():
    parser = ArgumentParser()
    parser.add_argument('-p', '--image_path', required=True)
    parser.add_argument('-s', '--save_file_path', required=True)
    parser.add_argument('-n', '--generate_num', required=True, type=int)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


if __name__ == "__main__":
    args = _get_args()

    if args.generate_num < 2:
        raise ValueError('生成数は2以上にしてください')

    if not Path(args.save_file_path).parent.exists():
        raise ValueError('フォルダが存在しません')

    if Path(args.save_file_path).suffix != '.csv':
        raise ValueError('拡張子はcsvにしてください')

    image_enhancer = ImageEnhancer(args.image_path)
    parameter_list = \
        generate_image_parameter_list(enhance_name_list, args.generate_num)

    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)

    game = TournamentGame(parameter_list)
    game.logger.addHandler(stream_handler)

    root = Tk()
    canvas = CompareCanvasGroupFrame(root, image_enhancer, game,
                                     args.save_file_path)
    canvas.logger.addHandler(stream_handler)
    canvas.pack()

    canvas.disp_enhanced_image()

    root.attributes('-topmost', True)
    root.lift()
    root.focus_force()

    root.mainloop()
