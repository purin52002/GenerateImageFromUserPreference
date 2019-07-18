import tkinter as tk
from tkinter import ttk

class ScoringWayAsker(tk.Toplevel):
    def __init__(self, master:ttk.Frame):
        super(ScoringWayAsker, self).__init__(master)

        self.tournament_button = ttk.Button(self, text='tournament')
        self.competition_button = ttk.Button(self, text='competition')

        self.tournament_button.pack(padx=10, pady=10, side=tk.LEFT)
        self.competition_button.pack(padx=10, pady=10, side=tk.LEFT)

        self.focus_force()
        self.transient(master)
        self.grab_set()
        self.resizable(0,0)

        self.title('select way to score')

if __name__ == "__main__":
    root = tk.Tk()
    ScoringWayAsker(root)
    root.mainloop()
