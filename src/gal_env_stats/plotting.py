"""
    Filename: plotting.py
    Author: Ericka Florio
    Description: Shared plotting and IO helpers used across the analysis modules.
"""

import os

import matplotlib.pyplot as plt


def make_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


# All generated figures are written under this directory, relative to the
# current working directory.
OUTPUT_DIR = "output/plots"


def save_figure(name, fig=None):
    """
        Save a figure (by file name) into OUTPUT_DIR, creating the directory if
        needed, then close it. If fig is None the current pyplot figure is used.
    """
    path = os.path.join(OUTPUT_DIR, name)
    make_directory(os.path.dirname(path) or ".")
    if fig is None:
        plt.savefig(path)
        plt.close()
    else:
        fig.savefig(path)
        plt.close(fig)
