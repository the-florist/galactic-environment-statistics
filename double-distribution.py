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

print("Plotting the double distribution.")
if pms.plot_dimension != 1 and pms.plot_dimension != 2:
    raise ValueError("double-distribution.py : plot_dimension impossible or is not implemented.")

beta_vals = np.logspace(np.log10(pms.beta_min), np.log10(pms.beta_max), pms.num_beta)
rho_vals = np.logspace(pms.rho_tilde_min, pms.rho_tilde_max, pms.num_rho)

print("-----\nDomain: ")
print(rf"Mass range: ({pms.beta_min:.1e}, {pms.beta_max:.1e})")
print("Density contrast range: ("+str(pms.rho_tilde_min)+", "+str(pms.rho_tilde_max)+")")

if pms.plot_dimension == 2:
    # Create meshgrid for m and delta_l
    BTS, RHOS = np.meshgrid(beta_vals, rho_vals, indexing='ij')

    print("Starting loop over mass/contrast domain...")

    # Compute joint pdf using dn
    Z = np.zeros_like(BTS)
    for i in range(pms.num_beta):
        for j in range(pms.num_rho):
            try:
                Z[i, j] = func.dn(BTS[i, j], RHOS[i, j])
            except Exception:
                Z[i, j] = 0

    # Marginals
    marginal_m = np.trapezoid(Z, rho_vals, axis=1)
    marginal_dl = np.trapezoid(Z, beta_vals, axis=0)

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)

    ax_joint = fig.add_subplot(gs[1:,:3])
    ax_marg_m = fig.add_subplot(gs[0,:3], sharex=ax_joint)
    ax_marg_dl = fig.add_subplot(gs[1:,3], sharey=ax_joint)

    # Joint pdf heatmap
    c = ax_joint.pcolormesh(beta_vals, rho_vals, Z.T, shading='auto', cmap='viridis')
    ax_joint.set_xlabel(r'$\beta$')
    ax_joint.set_ylabel(r'$\tilde{\rho}$')
    ax_joint.set_xscale('log')
    ax_joint.set_yscale('log')

    # Marginal for m
    ax_marg_m.plot(beta_vals, marginal_m, color='tab:blue')
    ax_marg_m.set_xscale('log')
    ax_marg_m.set_ylabel(r'Marginal ($\beta$)')
    ax_marg_m.tick_params(axis='y', labelbottom=False)
    ax_marg_m.grid(True, which='both', ls='--', alpha=0.3)

    # Marginal for delta_l
    ax_marg_dl.plot(marginal_dl, rho_vals, color='tab:orange')
    ax_marg_dl.set_xlabel(r'Marginal ($\tilde{\rho}$)')
    ax_marg_dl.tick_params(axis='y', labelleft=False)
    ax_marg_dl.grid(True, which='both', ls='--', alpha=0.3)
    ax_marg_dl.xaxis.get_offset_text().set_x(1.2)  # Move right
    ax_marg_dl.xaxis.get_offset_text().set_y(-0.15)  # Move down a bit

    # Remove tick labels on shared axes
    plt.setp(ax_marg_m.get_xticklabels(), visible=False)
    plt.setp(ax_marg_dl.get_yticklabels(), visible=False)

    ax_joint.set_title(r'Joint PDF of $\beta$ and $\tilde{\rho}$ with marginals')
    plt.savefig("plots/joint-pdf.pdf")
    plt.close()

elif pms.plot_dimension == 1:
    """
        Calculate the PDF at slices of beta, and plot alongside the mode and stdev of the mode.
    """

    beta_slices = np.array([1]) * (pms.beta_max - pms.beta_min) + pms.beta_min # , 0.25, 0.5, 0.75, 1
    for beta in beta_slices:
        print("Starting PDF calculation for mass %.2E..." % beta)

        PDF = [func.dn(beta, rho) for rho in rho_vals]
        mode, stdev = func.pdf_sample_expectation(PDF, rho_vals)

        # print(func.CDF(2, beta))
        # IQRL, IQRH = func.IQR(beta, mode - stdev, mode + stdev)

        # print("Sample mode: %.5E" % mode)
        # print("Sample variance from the mode: %.5E" % stdev)
        # print(f"Initial guess for IQR: [{mode - stdev}, {mode + stdev}]")
        # print(f"Predicted IQR: [{IQRL}, {IQRH}]")
        # print("---------")

        CDF = [func.CDF(rho, beta) for rho in rho_vals]
        # print(CDF[:10])
        # print(rho_vals[0], rho_vals[-1])
        # func.CDF(rho_vals[-1], pms.beta_max)
        # func.CDF(rho_vals[0], pms.beta_min)
        # print(func.CDF(rho_vals[-1], pms.beta_max), func.CDF(rho_vals[0], pms.beta_min))

        plt.plot(rho_vals, CDF, label=rf"$\beta$ = {beta:.2}")
        plt.plot(rho_vals, PDF, label=rf"$\beta$ = {beta:.2}")
        # plt.plot(mode, func.dn(beta, mode), 'ro', label='_nolegend_')
        # plt.errorbar(mode, func.dn(beta, mode), xerr=mode_stdev, fmt='r.', markersize=10)
        
        # Add IQR as a horizontal line segment centered at the mode
        # plt.hlines(func.dn(beta, mode), IQRL, IQRH, colors='red', linestyles='-', lw=2, label=f"IQR ($\\beta$={beta:.2f})")

        # Plot the IQR as vertical lines at IQRL and IQRH
        # plt.vlines([IQRL, IQRH], ymin=0, ymax=func.dn(beta, mode), colors='red', linestyles='dashed', lw=1.5, label='_nolegend_')
    
    plt.xlabel(r"$\tilde{\rho}$")
    plt.ylabel(r"$P_n$")
    plt.xscale("log")
    plt.title(r"CDF slices along $\beta$")
    plt.grid(True)
    plt.legend()

    plt.savefig("plots/joint-cdf-slice.pdf")
    plt.close()

"""
num_deltas = [25, 50, 100, 200]
if pms.run_convergence_tests:
    func.convergence_test(func.dn, func.pdf_analytic_expectation, func.pdf_sample_expectation, 
                            m_min, pms.beta_dd, delta_l_min, delta_l_max, num_deltas)
"""

print("Printed density profile.")