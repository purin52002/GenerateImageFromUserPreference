import tkinter as tk
from tkinter import ttk, filedialog
import os
from PIL import Image, ImageTk

class SelectPathFrame(ttk.Frame):
    def __init__(self, master:ttk.Frame, label_name:str):
        super(SelectPathFrame, self).__init__(master)
        self._init_style()

        self.label = ttk.Label(self, text=label_name, style='select_path.TLabel')
        self.path_var = tk.StringVar(self)
        self.entry = ttk.Entry(self, textvariable=self.path_var)
        self._make_select_path_button()

        self.label.pack(side=tk.LEFT, padx=5)
        self.entry.pack(side=tk.LEFT, padx=5)
        self.select_path_button.pack(side=tk.LEFT, padx=5)

    def _init_style(self):
        style = ttk.Style()
        style.configure('select_path.TLabel', font=('Helvetica', 15))
        # style.configure('select_path.TLabel', font=('', 20))

    def _make_select_path_button(self):
        abs_dir = os.path.abspath(os.path.dirname(__file__))
        image_path = os.path.join(abs_dir, 'resource', 'folder.png')
        self.select_button_image = ImageTk.PhotoImage(Image.open(image_path))
        self.select_path_button = ttk.Button(self, text='dir', image=self.select_button_image)

class SelectImagePathFrame(ttk.Frame):
    def __init__(self, master:ttk.Frame):
        super(SelectImagePathFrame, self).__init__(master)
        self.select_path_frame = SelectPathFrame(self, 'image')
        self.select_path_frame.select_path_button.bind('<1>', self.click_select_path_button)
        self.select_path_frame.pack()

    def click_select_path_button(self, event):
        file_type = [('', '*.png;*.jpg')]
        init_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../')
        self.select_path_frame.path_var.set(filedialog.askopenfilename(filetypes=file_type, initialdir=init_dir))

class SelectSavePathFrame(ttk.Frame):
    def __init__(self, master:ttk.Frame):
        super(SelectSavePathFrame, self).__init__(master)
        self.select_path_frame = SelectPathFrame(self, 'save')
        self.select_path_frame.select_path_button.bind('<1>', self.click_select_path_button)
        self.select_path_frame.pack()

    def click_select_path_button(self, event):
        file_type = [('', '*.csv')]
        init_dir = os.path.abspath(os.path.dirname(__file__))
        self.select_path_frame.path_var.set(filedialog.asksaveasfilename(filetypes=file_type, initialdir=init_dir))
