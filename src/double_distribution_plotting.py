"""
    Filename: double_distribution_plotting.py
    Author: Ericka Florio
    Created: 11 Sept 2025
    Description: Plotting routine for the joint double distribution for the 
            number density of objects 
            of mass m with local overdensity delta_l, 
            as derived in Pavlidou and Fields 2005.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.double_distribution_calculations import DoubleDistributionCalculations
import util.parameters as pms
import util.functions as func
import util.double_distribution_functions as ddfunc
from util.Newton_method import NewtonsMethod

class DoubleDistributionPlots(DoubleDistributionCalculations):
    m_norm = 1e14
    def __init__(self):
        ...
    def plot_heatmap(self, ddc:DoubleDistributionCalculations, fname):
        if pms.verbose:
            print("Starting heatmap plot.")
        
        b = np.abs(ddc.bvs - pms.beta_heuristic).argmin()

        # # Marginals
        marginal_m = ddc.PDF[b].sum(axis=0)
        marginal_rho = ddc.PDF[b].sum(axis=1)

        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)

        ax_joint = fig.add_subplot(gs[1:,:3])
        ax_marg_m = fig.add_subplot(gs[0,:3], sharex=ax_joint)
        ax_marg_dl = fig.add_subplot(gs[1:,3], sharey=ax_joint)

        # Joint pdf heatmap
        c = ax_joint.pcolormesh(ddc.rvs, ddc.mvs / self.m_norm, ddc.PDF[b].T, 
                                shading='auto', cmap='viridis')
        ax_joint.set_xlabel(r'$\rho/\rho_m$')
        ax_joint.set_ylabel(r'$m\ [10^{14} M_{\odot}]$')

        ax_joint.ticklabel_format(style='plain')

        # Make mass-axis (y) scientific offset (e.g., 1e15) vertical on the left
        y_off = ax_joint.yaxis.get_offset_text()
        y_off.set_rotation(90)
        y_off.set_verticalalignment('bottom')
        y_off.set_horizontalalignment('right')
        y_off.set_x(-0.1)
        y_off.set_clip_on(False)

        # Marginal for m
        ax_marg_m.plot(ddc.rvs, marginal_rho, color='tab:blue')
        ax_marg_m.set_ylabel(r'Marginal ($\bar{\rho}$)')
        ax_marg_m.grid(True, which='both', ls='--', alpha=0.3)
        ax_marg_m.set_title(r'Joint PDF of $m$ and $\tilde{\rho}$ at'+
                            r'$\beta = $' + str(b), pad=12)

        # Marginal for delta_l
        ax_marg_dl.plot(marginal_m, ddc.mvs / self.m_norm, color='tab:orange')
        ax_marg_dl.set_xlabel(r'Marginal ($m$)')
        ax_marg_dl.grid(True, which='both', ls='--', alpha=0.3)

        # # Remove tick labels on shared axes
        plt.setp(ax_marg_m.get_xticklabels(), visible=False)
        plt.setp(ax_marg_dl.get_yticklabels(), visible=False)

        # Save the plot
        func.make_directory("plots")
        plt.savefig(fname)
        plt.close()