from tkinter import Frame, LEFT, StringVar, Label, BOTTOM
from ImageEnhancer.image_enhancer import ImageEnhancer

from random import sample as random_sample
from ScoredParamIO.scored_param_writer import write_scored_param

from frame import KeyPressableFrame, CompareCanvasFrame
from game import TournamentGame, GameWin

from logging import getLogger, INFO


class CompareCanvasGroupFrame(KeyPressableFrame):
    def __init__(self, master: Frame, image_enhancer: ImageEnhancer,
                 game: TournamentGame, save_file_path: str):
        super(CompareCanvasGroupFrame, self).__init__(master)
        self.canvas_width = 300
        self.canvas_height = 300

        self.left_canvas = \
            CompareCanvasFrame(self, self.canvas_width, self.canvas_height)
        self.right_canvas = \
            CompareCanvasFrame(self, self.canvas_width, self.canvas_height)

        self.left_canvas.pack(side=LEFT)
        self.right_canvas.pack(side=LEFT)
        self.select_num_value = StringVar(self)
        Label(self, textvariable=self.select_num_value) \
            .pack(side=BOTTOM)

        self.bind(self.LEFT_PRESS, self._select_left, '+')
        self.bind(self.RIGHT_PRESS, self._select_right, '+')

        self.image_enhancer = image_enhancer
        self.game = game
        self.save_file_path = save_file_path

        self.select_num_value.set(f'残り選択回数: {self.game.get_match_num}')

        self.logger = getLogger('ComparaCanvas')
        self.logger.setLevel(INFO)

        self.focus_set()

    def disp_enhanced_image(self):
        left_param, right_param = self.game.new_match()

        left_enhanced_image = \
            self.image_enhancer.org_enhance(left_param)
        right_enhanced_image = \
            self.image_enhancer.org_enhance(right_param)

        self.left_canvas.canvas.update_image(left_enhanced_image)
        self.right_canvas.canvas.update_image(right_enhanced_image)

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
            write_scored_param(self.game.scored_player_list, self.save_file_path)
            self.master.destroy()
        else:
            self.disp_enhanced_image()


    def _scored_tournament(self, selected_index: int):
        self.scored_param_list[selected_index]['score'] *= 2

    def _scored_competition(self, selected_index: int, selected_score: int,
                            other_score: int):
        if selected_score <= other_score:
            self.scored_param_list[selected_index]['score'] = other_score
            for index in range(len(self.scored_param_list)):
                if index == selected_index:
                    continue

                scored_param = self.scored_param_list[index]
                if selected_score <= scored_param['score'] <= other_score:
                    scored_param['score'] -= 1

    def _tournament(self, selected_index: int, selected_score: int,
                    other_score: int):
        self._scored_tournament(selected_index)

        if len(self.current_image_parameter_index_list) < 2:
            if len(self.current_image_parameter_index_list) == 0 and\
                    len(self.next_image_parameter_index_list) == 1:
                print('tournament complete')
                write_scored_param(self.scored_param_list, self.save_file_path)
                # self._write_scored_parameter_to_csv()
                self.master.destroy()
                return

            self.next_image_parameter_index_list.extend(
                self.current_image_parameter_index_list)

            self.current_image_parameter_index_list = random_sample(
                self.next_image_parameter_index_list,
                len(self.next_image_parameter_index_list))

            self.next_image_parameter_index_list.clear()

        self.disp_enhanced_image()
