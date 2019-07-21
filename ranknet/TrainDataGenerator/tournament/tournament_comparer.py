from comparer import CompareCanvasGroupFrame
from tkinter import Tk
from argparse import ArgumentParser
from game import TournamentGame
from logging import StreamHandler, DEBUG
from pathlib import Path


def _get_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--image_dir_path', required=True)
    parser.add_argument('-o', '--save_file_path', required=True)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


if __name__ == "__main__":
    args = _get_args()

    if not Path(args.save_file_path).parent.exists():
        raise ValueError('フォルダが存在しません')

    if Path(args.save_file_path).suffix != '.json':
        raise ValueError('拡張子はjsonにしてください')

    image_dir_path = Path(args.image_dir_path)
    idol_dir_list = list(map(str, image_dir_path.iterdir()))

    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)

    game = TournamentGame(idol_dir_list)

    root = Tk()
    canvas = CompareCanvasGroupFrame(root, game, args.save_file_path)
    canvas.logger.addHandler(stream_handler)
    canvas.pack()

    canvas.disp_image()

    root.attributes('-topmost', True)
    root.lift()
    root.focus_force()

    root.mainloop()
