from tkinter import Frame, Button
from GUI.image_canvas_frame import ImageCanvas


class CompareCanvasFrame(Frame):
    def __init__(self, master: Frame,
                 canvas_width: int, canvas_height: int):
        super(CompareCanvasFrame, self).__init__(master)

        self.canvas = ImageCanvas(self, canvas_width, canvas_height)
        self.button = Button(self, text='good', width=6,)
        self.canvas.pack()
        self.button.pack()


class KeyPressableFrame(Frame):
    LEFT_KEY = 'f'
    RIGHT_KEY = 'j'

    LEFT_PRESS = f'<KeyPress-{LEFT_KEY}>'
    RIGHT_PRESS = f'<KeyPress-{RIGHT_KEY}>'

    LEFT_RELEASE = f'<KeyRelease-{LEFT_KEY}>'
    RIGHT_RELEASE = f'<KeyRelease-{RIGHT_KEY}>'

    def __init__(self, master: Frame):
        super(KeyPressableFrame, self).__init__(master)

        self.is_left_press = False
        self.is_right_press = False

        self.bind_all(self.LEFT_RELEASE, self._release_left, '+')
        self.bind_all(self.RIGHT_RELEASE, self._release_right, '+')

    def _release_left(self, e):
        self.is_left_press = False

    def _release_right(self, e):
        self.is_right_press = False
