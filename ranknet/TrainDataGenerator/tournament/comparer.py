from tkinter import Frame, LEFT, StringVar, Label, BOTTOM
import random
from frame import KeyPressableFrame, CompareCanvasFrame
from game import TournamentGame, GameWin
from logging import getLogger, INFO
from pathlib import Path
from PIL import Image


class CompareCanvasGroupFrame(KeyPressableFrame):
    def __init__(self, master: Frame, game: TournamentGame,
                 save_file_path: str, canvas_size=(300, 300)):
        super(CompareCanvasGroupFrame, self).__init__(master)
        self.canvas_width, self.canvas_height = canvas_size

        self.left_canvas = \
            CompareCanvasFrame(self, canvas_size)
        self.right_canvas = \
            CompareCanvasFrame(self, canvas_size)

        self.left_canvas.pack(side=LEFT)
        self.right_canvas.pack(side=LEFT)
        self.select_num_value = StringVar(self)
        Label(self, textvariable=self.select_num_value) \
            .pack(side=BOTTOM)

        self.bind(self.LEFT_PRESS, self._select_left, '+')
        self.bind(self.RIGHT_PRESS, self._select_right, '+')

        self.game = game
        self.save_file_path = save_file_path

        self.select_num_value.set(f'残り選択回数: {self.game.get_match_num}')

        self.logger = getLogger('ComparaCanvas')
        self.logger.setLevel(INFO)

        self.focus_set()

    def disp_image(self):
        def get_random_image(idol_path: Path):
            image_path_list = list(idol_path.iterdir())
            return random.choice(image_path_list)

        left_idol_path, right_idol_path = self.game.new_match()

        left_idol_path = Path(left_idol_path)
        right_idol_path = Path(right_idol_path)

        left_image = Image.open(get_random_image(left_idol_path))
        right_image = Image.open(get_random_image(right_idol_path))

        self.left_canvas.canvas.update_image(left_image)
        self.right_canvas.canvas.update_image(right_image)

        self.logger.debug('disp')

    def _select_left(self, e):
        self.logger.debug('press left')
        if self.is_left_press:
            self.logger.debug('hold left')
            return

        self.is_left_press = True
        self.game.compete(GameWin.LEFT)

        self._select_any()

    def _select_right(self, e):
        self.logger.debug('press right')
        if self.is_right_press:
            self.logger.debug('hold right')
            return

        self.is_right_press = True

        self.game.compete(GameWin.RIGHT)

        self._select_any()

    def _select_any(self):
        self.select_num_value.set(f'残り選択回数: {self.game.get_match_num}')
        if self.game.is_complete:
            self.game.save_as_json(self.save_file_path)
            self.master.destroy()
        else:
            self.disp_image()
