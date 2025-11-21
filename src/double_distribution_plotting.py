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

from src.double_distribution_calculations import DoubleDistributionCalculations as DDC
import util.parameters as pms
import util.functions as func
import util.double_distribution_functions as ddfunc
from util.Newton_method import NewtonsMethod

class DoubleDistributionPlots:
    m_norm = 1e14
    def __init__(self, ddc:DDC):
        self.fig, self.ax = plt.subplots()
        self.b = np.abs(ddc.bvs - pms.beta_heuristic).argmin()
    
    def savefig(self, fname):
        self.fig.savefig(fname)
        plt.close(self.fig)

    def plot_rho_slices(self, ddc:DDC):

        for mi, m in enumerate(ddc.mvs):
            line, = self.ax.plot(ddc.rvs, ddc.PDF[self.b,:,mi], 
                                    label=rf"$m = {ddc.MS[self.b,1,mi]:.2e}$")
            plot_color = line.get_color()

            # if pms.plot_statistics:
            #     # Plot analytic mode
            #     plt.plot(a_modes[self.b,mi], 
            #              ddfunc.dn(a_modes[self.b,mi], m, beta_vals[self.b], 
            #              transform_pdf=True) / norm[self.b,:,mi], 'o', color='red', 
            #              label='__nolabel__')

            #     # Plot numeric mode
            #     plt.plot(n_modes[self.b,mi], ddfunc.dn(n_modes[self.b,mi], m, beta_vals[self.b], 
            #                                 transform_pdf=True) / norm[self.b,:,mi],
            #                                 "*", color="blue", label='__nolabel__')

            #     # Plot analytic median
            #     plt.plot(a_stats[0, b,mi], 
            #         ddfunc.dn(a_stats[0, b,mi], m, beta_vals[self.b], 
            #                     transform_pdf=True) / norm[self.b,:,mi], 
            #         'o', color='red', label='__nolabel__')
                
            #     # Plot numeric median
            #     plt.plot(n_median[self.b,mi], 
            #             ddfunc.dn(n_median[self.b,mi], m, beta_vals[self.b], transform_pdf=True) / norm[self.b,:,mi],
            #             "*", color="blue", label='__nolabel__')     

            #     # Create logical masks where the PDF lies inside the IQRs
            #     a_mask = np.logical_and(rho_vals >= a_stats[1, b,mi], rho_vals 
            #                             <= a_stats[2, b,mi]).tolist()
            #     n_mask = np.logical_and(rho_vals >= n_IQRl[self.b,mi], rho_vals 
            #                             <= n_IQRu[self.b,mi]).tolist()

            #     # Fill in the IQRs
            #     plt.fill_between(rho_vals, cond_PDF[self.b,:,mi], 0, where = a_mask, 
            #                      alpha=0.5, color=plot_color)
            #     plt.fill_between(rho_vals, cond_PDF[self.b,:,mi], 0, where = n_mask, 
            #                      alpha=0.5, color=plot_color)

            # if pms.plot_untransformed_PDF:
            #         # Evaluate the untransformed PDF on the grid
            #         cond_PDF_nt, norm_nt = ddc.calc_PDF(False, pms.default_gamma)
            #         n_mode_nt, = ddc.n_stats()

            #         # Find the analytic and numeric most probable mode, untransformed
            #         a_mode_no_transform = ddc.a_stats(False)

            #         # Plot the untransformed PDF
            #         plt.plot(rho_vals, cond_PDF_nt[self.b,:,mi], color=plot_color, 
            #                 linestyle="--", label=rf"__nolabel__")

            #         if pms.plot_statistics:
            #             # Plot the analytic mode, untransformed
            #             plt.plot(a_mode_no_transform[self.b,mi], 
            #                 ddfunc.dn(a_mode_no_transform[self.b,mi], m, beta_vals[self.b],
            #                 transform_pdf=False) / norm_nt[self.b,:,mi], 
            #                 'o', color='red', label='__nolabel__')

            #             # Plot the numeric untransformed mode
            #             plt.plot(n_mode_nt[self.b,mi], 
            #                 ddfunc.dn(n_mode_nt[self.b,mi], m, beta_vals[self.b], 
            #                 transform_pdf=False) / norm_nt[self.b,:,mi],
            #                 "*", color="blue", label='__nolabel__')
                
            #     if pms.verbose:
            #         print(f"Plot finished for mass = {m:.2E}")

    def plot_heatmap(self, ddc:DDC, fname):
        if pms.verbose:
            print("Starting heatmap plot.")

        # # Marginals
        marginal_m = ddc.PDF[self.b].sum(axis=0)
        marginal_rho = ddc.PDF[self.b].sum(axis=1)

        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)

        ax_joint = fig.add_subplot(gs[1:,:3])
        ax_marg_m = fig.add_subplot(gs[0,:3], sharex=ax_joint)
        ax_marg_dl = fig.add_subplot(gs[1:,3], sharey=ax_joint)

        # Joint pdf heatmap
        c = ax_joint.pcolormesh(ddc.rvs, ddc.mvs / self.m_norm, ddc.PDF[self.b].T, 
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
                            r'$\beta = $' + str(self.b), pad=12)

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