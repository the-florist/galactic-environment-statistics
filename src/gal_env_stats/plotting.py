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


def save_figure(path, fig=None):
    """
        Ensure the target directory exists, then save and close the figure.
        If fig is None the current pyplot figure is used.
    """
    make_directory(os.path.dirname(path) or ".")
    if fig is None:
        plt.savefig(path)
        plt.close()
    else:
        fig.savefig(path)
        plt.close(fig)
