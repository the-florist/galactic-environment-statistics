"""
    Filename: double-distribution.py
    Author: Ericka Florio
    Created: 11 Sept 2025
    Description: Plots the joint double distribution for the number density of objects 
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
from src.double_distribution_calculations import DoubleDistributionCalculations
from src.double_distribution_plotting import DoubleDistributionPlots

if pms.plot_dimension != 1 and pms.plot_dimension != 2:
    raise ValueError("double-distribution.py : plot_dimension impossible" 
                     "or is not implemented.")

beta_vals = np.linspace(pms.beta_min, pms.beta_max, pms.num_beta)
rho_vals = np.linspace(pms.rho_tilde_min, pms.rho_tilde_max, pms.num_rho)
mass_vals = np.linspace(pms.mass_min, pms.mass_max, pms.num_mass) # np.array([1.3, 5, 17]) * 1e14  # np.array([pms.M_200])
gamma_slices = np.array([0.4]) # np.array([0.4, 0.5, 0.6]) # np.linspace(pms.gamma_min, pms.gamma_max, pms.num_gamma)


def run():
    # Create meshgrid with axes ordered as (beta, rho, mass)
    BTS, RHOS, MS = np.meshgrid(beta_vals, rho_vals, mass_vals, indexing='ij')

    if pms.verbose:
        print(f"Starting {pms.plot_dimension}D plot generation...")

    if pms.plot_dimension == 2:
        """
            Calculate the normalised joint PDF and plot it.
        """

        ddc = DoubleDistributionCalculations()
        ddp = DoubleDistributionPlots(ddc)
        ddc.calc_PDF(True, pms.default_gamma)
        ddp.plot_heatmap(ddc, "plots/joint-pdf.pdf")

    elif pms.plot_dimension == 1:
        """
            Calculate the conditional PDF at slices of beta, and plot alongside the mode 
            and stdev of the mode.
        """
        ddc = DoubleDistributionCalculations()
        ddp = DoubleDistributionPlots(ddc)

        if pms.slice_in_rho:
            # Numerically construct the conditional PDF
            cond_PDF, norm = ddc.calc_PDF(True, pms.default_gamma)

            # Find the closest beta to our heuristic value
            b = np.abs(beta_vals - pms.beta_heuristic).argmin()

            # Find the numerical modes, and variances rooted at those modes
            n_modes, n_stdevs, n_quants = ddc.n_stats()

            a_modes, a_stats = ddc.a_stats(True)

            for mi, m in enumerate(ddc.mvs):
                ddp.plot_rho_slice(mi)
                if pms.plot_statistics:
                    ddp.plot_a_stats(mi, True)
            
            ddp.savefig("plots/joint-pdf-slice.pdf")
            
            # Finish plot of PDF slices
            # plt.xlabel(r"$\tilde{\rho}$")
            # plt.ylabel(r"$P_n$")
            # plt.title(r"PDF slices along mass")
            # plt.grid(True)
            # plt.legend()
            # plt.savefig("plots/joint-pdf-slice.pdf")
            # plt.close()

        else:
            print("Starting most probable profile vs. mass plot...")
            for gi, g in enumerate(gamma_slices):
                # Numerically construct the conditional PDF
                cond_PDF, norm = ddc.calc_PDF(True, g=g)
                
                # Calculate the numerical statistics
                n_modes, n_stdevs, n_quants = ddc.n_stats()

                a_modes, a_stats = ddc.a_stats(True, g=g)
                
                # Plot analytic and numeric mode
                line, = plt.plot(beta_vals, a_modes[:, gi], 
                                label=rf"m={mass_vals[gi]:.2E}, $\gamma$={g:.2E}")
                mass_color = line.get_color()
                line, = plt.plot(beta_vals, n_modes[:, gi], 
                                color=mass_color, linestyle="--")

                # Plot analytic and numeric median
                plt.plot(beta_vals, a_stats[0, :, gi], linestyle="-.", color=mass_color)
                plt.plot(beta_vals, n_quants[0, :, gi], label="__nolabel__", 
                         color=mass_color, linestyle="dotted")

                # Plot analytic and numeric IQR
                plt.fill_between(beta_vals, a_stats[1, :, gi], a_stats[2, :, gi], 
                                 alpha=0.5, label="analytic IQR")
                plt.fill_between(beta_vals, n_quants[1, :, gi], n_quants[2, :, gi], 
                                 alpha=0.3, label="numeric IQR")

            plt.xlabel(r"$\beta$")
            plt.ylabel(r"$\hat{\rho}$")
            plt.title(r"Most probale profile vs. $\beta$")
            plt.grid(True)
            plt.legend()
            plt.savefig("plots/mpp-beta-scaling-with-iqrs.pdf")
            plt.close()


        if pms.plot_rho_derivative:
            def rho_derivative(rho):
                delta_c = ddfunc.delta_c_0(1) * func.D(1) / func.D(1)
                return pow(rho, (-1 - 1/delta_c))

            # Plot rho_derivative using rho_vals
            y_vals = [rho_derivative(r) for r in rho_vals]
            plt.figure()
            plt.plot(rho_vals, y_vals)
            plt.xlabel(r"$\bar{\rho}$")
            plt.ylabel(r"$d \tilde{\delta}_l / d \bar{\rho}$")
            plt.grid(True, which='both', ls='--', alpha=0.3)
            plt.title(r"rho derivative: $\bar{\rho}^{-1/\tilde{\delta}_{c}-1}$")
            func.make_directory("plots")
            plt.savefig("plots/rho-derivative.pdf")
            plt.close()

        