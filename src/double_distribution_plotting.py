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

class DoubleDistributionPlots:
    m_norm = 1e14
    plot_colors = []

    def __init__(self, ddc_in:DDC):
        self.ddc = ddc_in
        self.fig, self.ax = plt.subplots()
        self.b = np.abs(self.ddc.bvs - pms.beta_heuristic).argmin()
        
    """
        Helper functions, for saving and formatting plots.
    """
    
    def format_plot(self, title, x_label, y_label):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.legend()
    
    def save_plot(self, fname):
        func.make_directory("plots")
        self.fig.savefig(fname)
        plt.close(self.fig)

    """
        Plotting routines for beta slicing the PDF.
    """
    
    def plot_stat_slice(self, x, stat, gi, args):
        line, = self.ax.plot(x, stat[:,gi], **args)
        return line.get_color()
    
    def plot_beta_slices(self, gi):
        print(f"Plotting at {self.ddc.gamma_slices[gi]}...")
        
        a_mode_args = dict(label=(rf"m={self.ddc.mvs[gi]:.2E}, "+
                rf"$\gamma$={self.ddc.gamma_slices[gi]:.2E}"))
        mass_color = self.plot_stat_slice(self.ddc.bvs, self.ddc.a_mode, gi, a_mode_args)
        
        n_mode_args = dict(color=mass_color, linestyle="--")
        self.plot_stat_slice(self.ddc.bvs, self.ddc.n_mode, gi, n_mode_args)

        a_median_args = dict(color=mass_color, linestyle="-.")
        self.plot_stat_slice(self.ddc.bvs, self.ddc.a_quantiles[0], gi, a_median_args)

        n_median_args = dict(color=mass_color, linestyle="dotted")
        self.plot_stat_slice(self.ddc.bvs, self.ddc.n_quantiles[0], gi, n_median_args)

        if len(self.ddc.gamma_slices) == 1:
            plt.fill_between(self.ddc.bvs, self.ddc.a_quantiles[1, :, gi], 
                             self.ddc.a_quantiles[2, :, gi], alpha=0.5,
                             label="analytic IQR")
            plt.fill_between(self.ddc.bvs, self.ddc.n_quantiles[1, :, gi], 
                             self.ddc.n_quantiles[2, :, gi], alpha=0.5,
                             label="numeric IQR")
    
    """
        Plotting routines for rho slicing the PDF.
    """
    
    def plot_pdf_slice(self, x, mi, args):
        line, = self.ax.plot(x, self.ddc.PDF[self.b,:,mi], **args)
        return line.get_color()    
             
    def plot_rho_slice(self, mi, transf):
        if transf:
            plot_args = dict(label=rf"$m = {self.ddc.MS[self.b,1,mi]:.2e}$")
        else:
            plot_args = dict(linestyle='dashed', color=self.plot_colors[mi])
            
        color = self.plot_pdf_slice(self.ddc.rvs, mi, plot_args)
        
        if transf:
            self.plot_colors.append(color)
    
    def plot_point(self, rho, mi, transf, shape, color):
        plt.plot(rho, ddfunc.dn(rho, self.ddc.mvs[mi], self.ddc.bvs[self.b], 
                transform_pdf=transf) / self.ddc.norm[self.b,:,mi], 
                shape, color=color)

    def plot_quantile_mask(self, x, y, quant, bi, mi):
        mask = np.logical_and(self.ddc.rvs >= quant[1], 
                              self.ddc.rvs <= quant[2]).tolist()
        plt.fill_between(x, y, 0, where = mask, alpha=0.5, color=self.plot_colors[mi])

    def plot_a_stats(self, mi, transf):
        self.plot_point(self.ddc.a_mode[self.b, mi], mi, transf, 'o', 'red')
        self.plot_point(self.ddc.n_mode[self.b, mi], mi, transf, '*', 'blue')

        if transf:
            self.plot_point(self.ddc.n_quantiles[0, self.b, mi], mi, transf, 'o', 'red')
            self.plot_point(self.ddc.a_quantiles[0, self.b, mi], mi, transf, '*', 'blue') 

            self.plot_quantile_mask(self.ddc.rvs, self.ddc.PDF[self.b,:,mi], 
                                    self.ddc.n_quantiles[:, self.b, mi], 
                                    self.b, mi)
            
            self.plot_quantile_mask(self.ddc.rvs, self.ddc.PDF[self.b,:,mi], 
                                    self.ddc.a_quantiles[:, self.b, mi], 
                                    self.b, mi)

    
    """
        Misc. plotting routines, such as the heatmap and rho derivative plots.
    """

    def plot_rho_derivative(self):
        # Plot rho_derivative using rho_vals
        y_vals = [self.ddc.rho_derivative(r) for r in self.ddc.rvs]
        plt.plot(self.ddc.rvs, y_vals)
        plt.yscale('log')
        self.format_plot(r"rho derivative: $\bar{\rho}^{-1/\tilde{\delta}_{c}-1}$",
                        r"$\bar{\rho}$", r"$d \tilde{\delta}_l / d \bar{\rho}$")
        self.save_plot("plots/rho-derivative.pdf")
    
    def plot_heatmap(self, fname):
        if pms.verbose:
            print("Starting heatmap plot.")

        # # Marginals
        marginal_m = self.ddc.PDF[self.b].sum(axis=0)
        marginal_rho = self.ddc.PDF[self.b].sum(axis=1)

        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)

        ax_joint = fig.add_subplot(gs[1:,:3])
        ax_marg_m = fig.add_subplot(gs[0,:3], sharex=ax_joint)
        ax_marg_dl = fig.add_subplot(gs[1:,3], sharey=ax_joint)

        # Joint pdf heatmap
        c = ax_joint.pcolormesh(self.ddc.rvs, self.ddc.mvs / self.m_norm, self.ddc.PDF[self.b].T, 
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
        ax_marg_m.plot(self.ddc.rvs, marginal_rho, color='tab:blue')
        ax_marg_m.set_ylabel(r'Marginal ($\bar{\rho}$)')
        ax_marg_m.grid(True, which='both', ls='--', alpha=0.3)
        ax_marg_m.set_title(r'Joint PDF of $m$ and $\tilde{\rho}$ at'+
                            r'$\beta = $' + str(self.b), pad=12)

        # Marginal for delta_l
        ax_marg_dl.plot(marginal_m, self.ddc.mvs / self.m_norm, color='tab:orange')
        ax_marg_dl.set_xlabel(r'Marginal ($m$)')
        ax_marg_dl.grid(True, which='both', ls='--', alpha=0.3)

        # # Remove tick labels on shared axes
        plt.setp(ax_marg_m.get_xticklabels(), visible=False)
        plt.setp(ax_marg_dl.get_yticklabels(), visible=False)

        # Save the plot
        func.make_directory("plots")
        plt.savefig(fname)
        plt.close()