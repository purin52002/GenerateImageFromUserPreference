import tkinter
from tkinter import ttk, filedialog
import csv

from GUI.image_canvas_frame import ImageCanvasFrame
from ImageEnhancer.image_enhancer import ImageEnhancer


class ScoredCanvasFrame(ImageCanvasFrame):
    def __init__(self, master: ttk.Frame, canvas_width: int,
                 canvas_height: int, score: str):
        super(ScoredCanvasFrame, self).__init__(
            master, canvas_width, canvas_height)
        ttk.Label(self, text='score: %s' % (str(score))).pack()


if __name__ == "__main__":
    root = tkinter.Tk()
    root.attributes('-topmost', True)

    root.withdraw()
    root.lift()
    root.focus_force()

    disp_num = 4
    scored_param_list = []
    param_file = filedialog.askopenfile(
        'r', title='select scored_param data as csv',
        filetypes=[('scored param', ['.csv'])])

    if not param_file:
        exit()

    read_dict = csv.DictReader(param_file, delimiter=",", quotechar='"')
    key_list = read_dict.fieldnames

    scored_param_list = \
        sorted([dict(zip(row.keys(), map(float, row.values())))
                for row in read_dict], key=lambda x: -x['score'])[:disp_num]
    # print(scored_param_list)

    image_path = filedialog.askopenfilename(
        title='select enhance image', filetypes=[('image', ['.jpg', '.png'])])
    if not image_path:
        exit()

    image_enhancer = ImageEnhancer(image_path)

    root.deiconify()
    root.attributes('-topmost', True)

    for scored_param in scored_param_list:
        canvas = ScoredCanvasFrame(root, 300, 300, scored_param['score'])
        canvas.pack(side=tkinter.LEFT)
        canvas.update_image(image_enhancer.enhance(scored_param))

    root.mainloop()
