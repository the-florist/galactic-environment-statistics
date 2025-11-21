"""
    Filename: double_distribution_impl.py
    Author: Ericka Florio
    Created: 21 Nov. 2025
    Description: Declaration of a class that 
            Plots the joint double distribution for the number density of objects 
            of mass m with local overdensity delta_l, 
            as derived in Pavlidou and Fields 2005.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import util.parameters as pms
import util.functions as func
import util.double_distribution_functions as ddfunc
from util.Newton_method import NewtonsMethod

class DoubleDistribution:
    def __init__(self, plot_params, calc_params):