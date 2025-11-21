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

class DoubleDistributionCalculations:
    def __init__(self):
        bvs = np.linspace(pms.beta_min, pms.beta_max, pms.num_beta)
        rvs = np.linspace(pms.rho_tilde_min, pms.rho_tilde_max, pms.num_rho)
        mvs = np.linspace(pms.mass_min, pms.mass_max, pms.num_mass) # np.array([pms.M_200]) # np.array([1.3, 5, 17]) * 1e14 # np.linspace(pms.mass_min, pms.mass_max, pms.num_mass)
        gamma_slices = np.array([0.4]) # np.array([0.4, 0.5, 0.6]) # np.linspace(pms.gamma_min, pms.gamma_max, pms.num_gamma)

        self.BTS, self.RHOS, self.MS = np.meshgrid(bvs, rvs, mvs, indexing='ij')

    def calc_PDF(self, transfm, g = pms.default_gamma):
        PDF = ddfunc.dn(self.RHOS, self.MS, self.BTS, transform_pdf=transfm, gamma=g)
        if pms.normalise_pdf:
            if pms.plot_dimension == 2:
                axs = (1, 2)
            elif pms.plot_dimension == 1:
                axs = 1
            
            norm = PDF.sum(axis=axs, keepdims=True)
            PDF /= norm
            if pms.verbose:
                print(f"PDF norm - 1: {abs(PDF.sum(axis=axs) - 1.).max()}")
                print(f"PDF max value: ", PDF.max(axis=axs))
        return PDF, norm
    