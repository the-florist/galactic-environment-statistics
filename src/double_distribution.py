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
gamma_slices = np.linspace(pms.gamma_min, pms.gamma_max, pms.num_gamma)


def run():
    # Create meshgrid with axes ordered as (beta, rho, mass)
    BTS, RHOS, MS = np.meshgrid(beta_vals, rho_vals, mass_vals, indexing='ij')

    if pms.verbose:
        print(f"Starting {pms.plot_dimension}D plot generation...")

    if pms.plot_dimension == 2:
        """
            Calculate and plot the normalised joint PDF.
        """

        ddc = DoubleDistributionCalculations()
        ddp = DoubleDistributionPlots(ddc)
        ddc.calc_PDF(True, pms.default_gamma)
        ddp.plot_heatmap("plots/joint-pdf.pdf")

    elif pms.plot_dimension == 1:
        """
            Calculate the conditional PDF at slices of rho or beta, and 
            plot alongside the mode, median and quantiles.
        """
        ddc = DoubleDistributionCalculations()
        ddp = DoubleDistributionPlots(ddc)

        if pms.slice_in_rho:
            # Find the closest beta to our heuristic value
            transform_pdf = True

            # Construct the conditional PDF and its statistics
            ddc.calc_PDF(transform_pdf, pms.default_gamma)
            ddc.n_stats()
            ddc.a_stats(transform_pdf)

            # Plot the full PDF
            for mi in range(len(ddc.mvs)):
                ddp.plot_rho_slice(mi, transform_pdf)
                if pms.plot_statistics:
                    ddp.plot_a_stats(mi, transform_pdf)

            # Construct and plot the untransformed PDF and its statistics
            if pms.plot_untransformed_PDF:
                transform_pdf = False
                ddc.calc_PDF(transform_pdf, pms.default_gamma)
                ddc.n_stats()
                ddc.a_stats(transform_pdf)

                for mi in range(len(ddc.mvs)):
                    ddp.plot_rho_slice(mi, transform_pdf)
                    if pms.plot_statistics:
                        ddp.plot_a_stats(mi, transform_pdf)

            # Finish plot of PDF slices
            ddp.format_plot(r"PDF slices along mass", r"$\tilde{\rho}$", r"$P_n$")
            ddp.save_plot("plots/joint-pdf-slice.pdf")

        else:
            print("Starting most probable profile vs. mass plot...")
            for gi, g in enumerate(gamma_slices):
                # Numerically construct the conditional PDF
                transform_pdf = True
                ddc.calc_PDF(transform_pdf, g=g)
                
                # Calculate the numerical statistics
                ddc.n_stats()
                ddc.a_stats(transform_pdf, g=g)

                # Plot the stats as a function of beta
                ddp.plot_beta_slices(gi)

            # Format and save the plot
            ddp.format_plot(r"Most probale profile vs. $\beta$", 
                            r"$\beta$", r"$\hat{\rho}$")
            ddp.save_plot("plots/mpp-scaling.pdf")


        if pms.plot_rho_derivative:
            ddc = DoubleDistributionCalculations()
            ddp = DoubleDistributionPlots(ddc)
            ddp.plot_rho_derivative()

        