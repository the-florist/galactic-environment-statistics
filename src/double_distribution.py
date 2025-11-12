"""
    Filename: double-distribution.py
    Author: Ericka Florio
    Created: 11 Sept 2025
    Description: Plots the joint double distribution for the number density of objects 
            of mass m with local overdensity delta_l, 
            as derived in Pavlidou and Fields 2005.
"""

from matplotlib import style
from matplotlib.lines import lineStyles
import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

        PDF = np.zeros((pms.num_beta_slices, pms.num_rho, pms.num_mass))

        # Set up the timer
        start = time()
        intv = 15
        last = start

        for i in range(pms.num_beta):
            for j in range(pms.num_rho):
                for k in range(pms.num_mass):
                    try:
                        PDF[i, j, k] = ddfunc.dn(RHOS[i, j, k], MS[i, j, k], BTS[i, j, k])
                    except Exception:
                        PDF[i, j, k] = 0

                now = time()
                if now - last > intv:
                    frac = ((j * pms.num_rho) + k + 1) / (pms.num_mass * pms.num_rho)
                    elapsed = now - start
                    print(f"{elapsed:.2g} sec. passed for beta = "+str(beta_vals[i])+f", {frac * 100}% finished...")
                    last = now

        # Normalise the joint PDF
        if pms.normalise_pdf:
            for i in range(pms.num_beta):
                norm = np.sum(PDF[i])
                PDF[i] /= norm
                print(f"PDF norm precision: {abs(np.sum(PDF[i]) - 1.):.15}")

        b = pms.beta_heuristic

        # Marginals
        marginal_dl = np.trapezoid(PDF[b], mass_vals, axis=1)
        marginal_m = np.trapezoid(PDF[b], rho_vals, axis=0)

        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)

        ax_joint = fig.add_subplot(gs[1:,:3])
        ax_marg_m = fig.add_subplot(gs[0,:3], sharex=ax_joint)
        ax_marg_dl = fig.add_subplot(gs[1:,3], sharey=ax_joint)

        # Joint pdf heatmap
        c = ax_joint.pcolormesh(rho_vals, mass_vals, PDF[b].T, shading='auto', 
                                                          cmap='viridis')
        ax_joint.set_xlabel(r'$\rho/\rho_m$')
        ax_joint.set_ylabel(r'$m$')
        # ax_joint.set_xscale('log')
        # ax_joint.set_yscale('log')

        # Make mass-axis (y) scientific offset (e.g., 1e15) vertical on the left
        ax_joint.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        y_off = ax_joint.yaxis.get_offset_text()
        y_off.set_rotation(90)
        y_off.set_verticalalignment('bottom')
        y_off.set_horizontalalignment('right')
        y_off.set_x(-0.1)
        y_off.set_clip_on(False)

        # Marginal for m
        ax_marg_m.plot(rho_vals, marginal_dl, color='tab:blue')
        # ax_marg_m.set_xscale('log')
        ax_marg_m.set_ylabel(r'Marginal ($\bar{\rho}$)')
        ax_marg_m.tick_params(axis='y', labelbottom=False)
        ax_marg_m.grid(True, which='both', ls='--', alpha=0.3)

        # Marginal for delta_l
        ax_marg_dl.plot(marginal_m, mass_vals, color='tab:orange')
        ax_marg_dl.set_xlabel(r'Marginal ($m$)')
        ax_marg_dl.tick_params(axis='y', labelleft=False)
        ax_marg_dl.grid(True, which='both', ls='--', alpha=0.3)
        ax_marg_dl.xaxis.get_offset_text().set_x(1.2)  # Move right
        ax_marg_dl.xaxis.get_offset_text().set_y(-0.15)  # Move down a bit

        # Remove tick labels on shared axes
        plt.setp(ax_marg_m.get_xticklabels(), visible=False)
        plt.setp(ax_marg_dl.get_yticklabels(), visible=False)

        ax_joint.set_title(r'Joint PDF of $m$ and $\tilde{\rho}$ at $\beta = $'+str(b))

        func.make_directory("plots")
        plt.savefig("plots/joint-pdf.pdf")
        plt.close()

    elif pms.plot_dimension == 1:
        """
            Calculate the conditional PDF at slices of beta, and plot alongside the mode 
            and stdev of the mode.
        """

        slices = np.array([1e14, 1e15]) # np.array([mass_vals[i] for i in range(0, int(pms.num_mass), 
                 #                                   round(pms.num_mass/5))])
        b = pms.beta_heuristic

        # Generate slice of PDF at beta_slice
        for m in slices:
            cond_PDF = []
            for r in rho_vals:
                cond_PDF.append(ddfunc.dn(r, m, b, transform_pdf=True))
            norm = (sum(cond_PDF) - cond_PDF[0])
            cond_PDF /= norm
            print(f"PDF norm precision: {abs(np.sum(cond_PDF) - 1.):.15}")

            cond_PDF_no_transform = []
            for r in rho_vals:
                cond_PDF_no_transform.append(ddfunc.dn(r, m, b, transform_pdf=False))
            norm_no_transform = (sum(cond_PDF_no_transform) - cond_PDF_no_transform[0])
            cond_PDF_no_transform /= norm_no_transform

            numeric_mode, numeric_stdev = ddfunc.pdf_sample_expectation(cond_PDF, rho_vals)
            analytic_mode_transformed = ddfunc.most_probable_rho_transformed(m, b)
            analytic_mode_no_transform = ddfunc.most_probable_rho(b, inc_mass_scaling=True, m=m)
            aIQRl, aIQRu = ddfunc.analytic_IQR(numeric_mode, numeric_stdev, b, m)
            nIQRl, nIQRu = ddfunc.numeric_IQR(cond_PDF, rho_vals)

            print("Numeric IQR estimate: ", ddfunc.numeric_CDF(cond_PDF, rho_vals, nIQRl), ddfunc.numeric_CDF(cond_PDF, rho_vals, nIQRu))
            print("Analytic IQR estimate: ", ddfunc.numeric_CDF(cond_PDF, rho_vals, aIQRl), ddfunc.numeric_CDF(cond_PDF, rho_vals, aIQRu))
            print("--------------------------------")

            line, = plt.plot(rho_vals, cond_PDF, label=rf"$m = {m:.2E}$")
            plot_color = line.get_color()
            plt.plot(rho_vals, cond_PDF_no_transform, color=plot_color, linestyle="--", label=rf"__nolabel__")

            plt.plot(analytic_mode_transformed, 
                    ddfunc.dn(analytic_mode_transformed, m, b, transform_pdf=True) / norm, 
                    'o', color='red', label='__nolabel__')
            plt.plot(analytic_mode_no_transform, 
                    ddfunc.dn(analytic_mode_no_transform, m, b, transform_pdf=False) / norm_no_transform, 
                    'o', color='red', label='__nolabel__')
            plt.plot(numeric_mode, 
                    ddfunc.dn(numeric_mode, m, b, transform_pdf=True) / norm,
                    "*", color="blue", label='__nolabel__')

            plt.axvline(x=aIQRl, color=plot_color, linestyle="-", label=rf"__nolabel__")
            plt.axvline(x=aIQRu, color=plot_color, linestyle="-", label=rf"__nolabel__")
            plt.axvline(x=nIQRl, color=plot_color, linestyle="-.", label=rf"__nolabel__")
            plt.axvline(x=nIQRu, color=plot_color, linestyle="-.", label=rf"__nolabel__")

            print(f"Plot finished for mass = {m:.2E}")
        
        # Finish plot of PDF slices
        plt.xlabel(r"$\tilde{\rho}$")
        plt.ylabel(r"$P_n$")
        # plt.xscale("log")
        plt.title(r"PDF slices along mass")
        plt.grid(True)
        plt.legend()
        plt.savefig("plots/joint-pdf-slice.pdf")
        plt.close()

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

        