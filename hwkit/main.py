"""Entry point: launches the GUI. Run with `python main.py`. Requirements are listet in Requirements.txt"""

import os

# aihwkit and PyTorch each bundle their own OpenMP runtime; on macOS loading both
# crashes the process unless this flag is set. Set it here (before anything imports
# torch) so the app runs even outside the conda env setup.sh configures. Harmless
# on Windows/Linux.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import tkinter as tk

from gui_frontend import App

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()