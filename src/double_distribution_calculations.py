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
        self.bvs = np.linspace(pms.beta_min, pms.beta_max, pms.num_beta)
        self.rvs = np.linspace(pms.rho_tilde_min, pms.rho_tilde_max, pms.num_rho)
        self.mvs = np.linspace(pms.mass_min, pms.mass_max, pms.num_mass) # np.array([pms.M_200]) # np.array([1.3, 5, 17]) * 1e14 # np.linspace(pms.mass_min, pms.mass_max, pms.num_mass)
        self.gamma_slices = np.array([0.4]) # np.array([0.4, 0.5, 0.6]) # np.linspace(pms.gamma_min, pms.gamma_max, pms.num_gamma)

        self.BTS, self.RHOS, self.MS = np.meshgrid(self.bvs, self.rvs, 
                                                   self.mvs, indexing='ij')

    def calc_PDF(self, transfm, g = pms.default_gamma):
        if pms.verbose:
            print("Starting PDF calculation.")
            
        self.PDF = ddfunc.dn(self.RHOS, self.MS, self.BTS, 
                             transform_pdf=transfm, gamma=g)
        if pms.normalise_pdf:
            if pms.plot_dimension == 2:
                axs = (1, 2)
            elif pms.plot_dimension == 1:
                axs = 1
            
            self.norm = self.PDF.sum(axis=axs, keepdims=True)
            self.PDF /= self.norm
            if pms.verbose:
                print(f"PDF norm - 1: {abs(self.PDF.sum(axis=axs) - 1.).max()}")
                print(f"PDF max value: ", self.PDF.max(axis=axs))
        # return PDF, norm
    