"""
    Filename: double-distribution.py
    Author: Ericka Florio
    Created: 11 Sept 2025
    Description: Plots the joint double distribution for the number density of objects 
            of mass m with local overdensity delta_l, 
            as derived in Pavlidou and Fields 2005.
"""

import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

import util.parameters as pms
import util.functions as func
import src.double_distribution_functions as ddfunc

if pms.plot_dimension != 1 and pms.plot_dimension != 2:
    raise ValueError("double-distribution.py : plot_dimension impossible" 
                     "or is not implemented.")

beta_vals = np.linspace(pms.beta_min, pms.beta_max, pms.num_beta)
rho_vals = np.linspace(pms.rho_tilde_min, pms.rho_tilde_max, pms.num_rho)
mass_vals = np.linspace(pms.mass_min, pms.mass_max, pms.num_mass)
# rho_vals = np.array([pow(10, r) for r in lin_rho_vals])
# dr = (pms.rho_tilde_max - pms.rho_tilde_min) / (pms.num_rho)
# db = (pms.beta_min - pms.beta_max) / (pms.num_beta)


def run():
    # Create meshgrid with axes ordered as (beta, rho, mass)
    BTS, RHOS, MS = np.meshgrid(beta_vals, rho_vals, mass_vals, indexing='ij')

    print(f"Starting {pms.plot_dimension}D plot generation...")

    if pms.plot_dimension == 2:
        """
            Calculate the normalised joint PDF and plot it.
        """

        PDF = ddfunc.dn(RHOS, MS, BTS)

        # Normalise the joint PDF
        if pms.normalise_pdf:
            PDF /= PDF.sum(axis=(1, 2), keepdims=True)
            print(f"PDF norm precision: {abs(PDF[0].sum() - 1.):.15}")
            print(f"PDF max value: ", PDF[0].max())

        b = np.abs(beta_vals - pms.beta_heuristic).argmin()

        # Marginals
        marginal_m = PDF[b].sum(axis=0)
        marginal_rho = PDF[b].sum(axis=1)

        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)

        ax_joint = fig.add_subplot(gs[1:,:3])
        ax_marg_m = fig.add_subplot(gs[0,:3], sharex=ax_joint)
        ax_marg_dl = fig.add_subplot(gs[1:,3], sharey=ax_joint)

        # Joint pdf heatmap
        c = ax_joint.pcolormesh(rho_vals, mass_vals / 1e14, PDF[b].T, shading='auto', 
                                                          cmap='viridis')
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
        ax_marg_m.plot(rho_vals, marginal_rho, color='tab:blue')
        ax_marg_m.set_ylabel(r'Marginal ($\bar{\rho}$)')
        ax_marg_m.grid(True, which='both', ls='--', alpha=0.3)
        ax_marg_m.set_title(r'Joint PDF of $m$ and $\tilde{\rho}$ at $\beta = $' + str(b), pad=12)

        # Marginal for delta_l
        ax_marg_dl.plot(marginal_m, mass_vals / 1e14, color='tab:orange')
        ax_marg_dl.set_xlabel(r'Marginal ($m$)')
        ax_marg_dl.grid(True, which='both', ls='--', alpha=0.3)

        # Remove tick labels on shared axes
        plt.setp(ax_marg_m.get_xticklabels(), visible=False)
        plt.setp(ax_marg_dl.get_yticklabels(), visible=False)


        func.make_directory("plots")
        plt.savefig("plots/joint-pdf.pdf")
        plt.close()

    elif pms.plot_dimension == 1:
        """
            Calculate the conditional PDF at slices of beta, and plot alongside the mode 
            and stdev of the mode.
        """

        if pms.slice_in_rho:
            mass_slices = np.array([1e14, 1e15]) # np.array([mass_vals[i] for i in range(0, int(pms.num_mass), 
                 #                                   round(pms.num_mass/5))])
            b = np.abs(beta_vals - pms.beta_heuristic).argmin()

            cond_PDF = ddfunc.dn(RHOS, mass_slices, b, transform_pdf=True)
            print(cond_PDF.shape)
            # if pms.normalise_pdf:
            #     cond_PDF /= cond_PDF.sum(axis=1)

            # Generate slice of PDF at beta_slice
            for m in slices:
                cond_PDF = []
                for r in rho_vals:
                    cond_PDF.append(ddfunc.dn(r, m, b, transform_pdf=True))
                norm = (sum(cond_PDF) - cond_PDF[0])
                cond_PDF /= norm
                if pms.verbose:
                    print(f"PDF norm precision: {abs(np.sum(cond_PDF) - 1.):.15}")

                line, = plt.plot(rho_vals, cond_PDF, label=rf"$m = {m:.2E}$")
                mass_plot_color = line.get_color()

                if pms.plot_untransformed_PDF:
                    cond_PDF_no_transform = []
                    for r in rho_vals:
                        cond_PDF_no_transform.append(ddfunc.dn(r, m, b, transform_pdf=False))
                    norm_no_transform = (sum(cond_PDF_no_transform) - cond_PDF_no_transform[0])
                    cond_PDF_no_transform /= norm_no_transform

                    plt.plot(rho_vals, cond_PDF_no_transform, color=mass_plot_color, linestyle="--", label=rf"__nolabel__")

                if pms.plot_statistics:
                    numeric_mode, numeric_stdev = ddfunc.pdf_sample_expectation(cond_PDF, rho_vals)
                    analytic_mode_transformed = ddfunc.most_probable_rho_transformed(m, b)
                    analytic_IQRl, analytic_IQRu, analytic_median = ddfunc.analytic_median_and_IQR(numeric_mode, numeric_stdev, b, m)
                    numeric_IQRl, numeric_IQRu, numeric_median = ddfunc.numeric_median_and_IQR(cond_PDF, rho_vals)

                    if pms.verbose:
                        print("Numeric IQR estimate: ", 
                            ddfunc.numeric_CDF(cond_PDF, rho_vals, numeric_IQRl), 
                            ddfunc.numeric_CDF(cond_PDF, rho_vals, numeric_IQRu))
                        print("Analytic IQR estimate: ", 
                            ddfunc.numeric_CDF(cond_PDF, rho_vals, analytic_IQRl), 
                            ddfunc.numeric_CDF(cond_PDF, rho_vals, analytic_IQRu))
                        print("Median estimates: ", numeric_median, analytic_median)
                        print("--------------------------------")

                    plt.plot(analytic_mode_transformed, 
                            ddfunc.dn(analytic_mode_transformed, m, b, transform_pdf=True) / norm, 
                            'o', color='red', label='__nolabel__')
                    plt.plot(numeric_mode, 
                            ddfunc.dn(numeric_mode, m, b, transform_pdf=True) / norm,
                            "*", color="blue", label='__nolabel__')
                    plt.plot(analytic_median, 
                            ddfunc.dn(analytic_median, m, b, transform_pdf=True) / norm, 
                            'o', color='red', label='__nolabel__')
                    plt.plot(numeric_median, 
                            ddfunc.dn(numeric_median, m, b, transform_pdf=True) / norm,
                            "*", color="blue", label='__nolabel__')

                    a_mask = np.logical_and(rho_vals >= analytic_IQRl, rho_vals 
                                            <= analytic_IQRu).tolist()
                    n_mask = np.logical_and(rho_vals >= numeric_IQRl, rho_vals 
                                            <= numeric_IQRu).tolist()

                    plt.fill_between(rho_vals, cond_PDF, 0, where = a_mask, 
                                     alpha=0.5, color=mass_plot_color)
                    plt.fill_between(rho_vals, cond_PDF, 0, where = n_mask, 
                                     alpha=0.5, color=mass_plot_color)

                    if pms.plot_untransformed_PDF:
                        analytic_mode_no_transform = ddfunc.most_probable_rho(b, inc_mass_scaling=True, m=m)
                        numeric_mode_no_transform, _ = ddfunc.pdf_sample_expectation(cond_PDF_no_transform, rho_vals)

                        plt.plot(analytic_mode_no_transform, 
                            ddfunc.dn(analytic_mode_no_transform, m, b, transform_pdf=False) / norm_no_transform, 
                            'o', color='red', label='__nolabel__')
                        plt.plot(numeric_mode_no_transform, 
                            ddfunc.dn(numeric_mode_no_transform, m, b, transform_pdf=False) / norm_no_transform,
                            "*", color="blue", label='__nolabel__')
                
                if pms.verbose:
                    print(f"Plot finished for mass = {m:.2E}")
            
            # Finish plot of PDF slices
            plt.xlabel(r"$\tilde{\rho}$")
            plt.ylabel(r"$P_n$")
            plt.title(r"PDF slices along mass")
            plt.grid(True)
            plt.legend()
            plt.savefig("plots/joint-pdf-slice.pdf")
            plt.close()

        else:
            print("Starting most probable profile vs. mass plot...")
            mass_slices = np.array([1]) * 1e14 #  5.0, 17
            gamma_slices = np.array([0.4]) # , 0.5, 0.6

            modes = np.zeros((pms.num_beta, 2))
            medians = np.zeros((pms.num_beta, 2))
            IQRs = np.zeros((pms.num_beta, 4))

            # Set up the timer
            start = time()
            intv = 15
            last = start

            for mi, m in enumerate(mass_slices):
                for bi, b in enumerate(beta_vals):
                    now = time()
                    if now - last > intv:
                        frac = bi / (pms.num_beta)
                        elapsed = now - start
                        print(f"{elapsed:.2g} sec. passed, {frac * 100}% finished...")
                        last = now

                    cond_PDF = []
                    for r in rho_vals:
                        cond_PDF.append(ddfunc.dn(r, m, b, transform_pdf=True, g=gamma_slices[mi]))
                    norm = (sum(cond_PDF) - cond_PDF[0])
                    cond_PDF /= norm

                    modes[bi, 0] = ddfunc.most_probable_rho_transformed(m, b, gamma=gamma_slices[mi])
                    modes[bi, 1], numeric_stdev = ddfunc.pdf_sample_expectation(cond_PDF, rho_vals)
                    IQRs[bi, 0], IQRs[bi, 1], medians[bi, 0] = ddfunc.analytic_median_and_IQR(modes[bi, 1], numeric_stdev, b, m, gamma=gamma_slices[mi])
                    IQRs[bi, 2], IQRs[bi, 3], medians[bi, 1] = ddfunc.numeric_median_and_IQR(cond_PDF, rho_vals)

                line, = plt.plot(beta_vals, modes[:, 0], label=rf"m={m:.2E}, $\gamma$={gamma_slices[mi]:.2E}")
                mass_color = line.get_color()
                plt.plot(beta_vals, modes[:, 1], label="__nolabel__", color=mass_color, linestyle="--")

                plt.plot(beta_vals, medians[:, 0], linestyle="-.", color=mass_color)
                plt.plot(beta_vals, medians[:, 1], label="__nolabel__", color=mass_color, linestyle="dotted")

                # plt.fill_between(mass_vals, IQRs[:, 0], IQRs[:, 1], alpha=0.5, label="analytic IQR")
                # plt.fill_between(mass_vals, IQRs[:, 2], IQRs[:, 3], alpha=0.5, label="numeric IQR")

            plt.xlabel(r"$\beta$")
            plt.ylabel(r"$\hat{\rho}$")
            plt.title(r"Most probale profile vs. $\beta$")
            plt.grid(True)
            plt.legend()
            plt.savefig("plots/mpp-beta-scaling.pdf")
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

        