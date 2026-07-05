"""
    Filename: functions.py
    Author: Ericka Florio
    Created: 8th September 2025
    Description: General-purpose IO helpers shared across the package. The
                 physics/maths routines have moved to the physics/ subpackage.
"""

import os


def make_directory(output_dir):
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
