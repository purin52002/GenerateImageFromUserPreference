from tkinter import Frame, Button, Canvas
from PIL import Image, ImageTk


class ImageCanvas(Canvas):
    def __init__(self, master: Frame, canvas_size: tuple):
        self.canvas_size = canvas_size
        canvas_width, canvas_height = canvas_size
        super(ImageCanvas, self).__init__(
            master, width=canvas_width, height=canvas_height)
        self.image_id = self.create_image(canvas_width//2, canvas_height//2)
        self.image = None

    def update_image(self, image: Image.Image):
        image = image.resize(self.canvas_size)
        self.image = ImageTk.PhotoImage(image)
        self.itemconfigure(self.image_id, image=self.image)


class CompareCanvasFrame(Frame):
    def __init__(self, master: Frame, canvas_size: tuple):
        super(CompareCanvasFrame, self).__init__(master)

        self.canvas = ImageCanvas(self, canvas_size)
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
