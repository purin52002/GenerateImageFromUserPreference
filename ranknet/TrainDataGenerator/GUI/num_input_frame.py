import tkinter as tk
from tkinter import ttk, simpledialog

class IntInputFrame(ttk.Frame):
    def __init__(self, master:ttk.Frame):
        super(IntInputFrame, self).__init__(master)

        self.data_num_var = tk.StringVar(self)
        self.data_num_var.trace('w', self._spin_changed)
        tk.Spinbox(self, textvariable=self.data_num_var, from_=2, to=1000).pack(side=tk.LEFT)

    def _spin_changed(self, *args):
        if self.data_num_var.get().isnumeric():
            s = self.data_num_var.get()
            i = int(s)
            self.data_num_var.set(i)
        else:
            self.data_num_var.set("2")

def input_int_value():
    root = tk.Tk()
    # root.withdraw()

    sub_win = tk.Toplevel()
    sub_win.geometry('300x200')
    input_frame = IntInputFrame(sub_win)
    input_frame.pack()
    ttk.Button(sub_win, text='ok', command=root.destroy).pack()
    sub_win.mainloop()

    root.destroy()
    return int(input_frame.data_num_var.get())

if __name__ == "__main__":
    print(input_int_value())