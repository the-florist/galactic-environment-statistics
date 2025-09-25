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

# Both independent vars. are unitless
beta_min, beta_max, num_beta = 1.3, 5, 50
rho_tilde_min, rho_tilde_max, num_rho = 0, 3, 50

beta_vals = np.logspace(np.log10(beta_min), np.log10(beta_max), num_beta)
rho_vals = np.logspace(rho_tilde_min, rho_tilde_max, num_rho)

print("-----\nDomain: ")
print(rf"Mass range: ({beta_min:.1e}, {beta_max:.1e})")
print("Density contrast range: ("+str(rho_tilde_min)+", "+str(rho_tilde_max)+")")

if pms.plot_dimension == 2:
    # Create meshgrid for m and delta_l
    BTS, RHOS = np.meshgrid(beta_vals, rho_vals, indexing='ij')

    print("Starting loop over mass/contrast domain...")

    # Compute joint pdf using dn
    Z = np.zeros_like(BTS)
    for i in range(num_beta):
        for j in range(num_rho):
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
    ax_joint.set_ylabel(r'$\rho/\bar{\rho}_m$')
    ax_joint.set_xscale('log')
    ax_joint.set_yscale('log')

    # Marginal for m
    ax_marg_m.plot(beta_vals, marginal_m, color='tab:blue')
    ax_marg_m.set_xscale('log')
    ax_marg_m.set_ylabel('Marginal\nPDF (m)')
    ax_marg_m.tick_params(axis='y', labelbottom=False)
    ax_marg_m.grid(True, which='both', ls='--', alpha=0.3)

    # Marginal for delta_l
    ax_marg_dl.plot(marginal_dl, rho_vals, color='tab:orange')
    ax_marg_dl.set_xlabel(r'Marginal\nPDF ($\delta_l$)')
    ax_marg_dl.tick_params(axis='y', labelleft=False)
    ax_marg_dl.grid(True, which='both', ls='--', alpha=0.3)

    # Remove tick labels on shared axes
    plt.setp(ax_marg_m.get_xticklabels(), visible=False)
    plt.setp(ax_marg_dl.get_yticklabels(), visible=False)

    ax_joint.set_title(r'Joint PDF of m and $\delta_l$ with marginals')
    plt.savefig("plots/joint-pdf.pdf")
    plt.close()

elif pms.plot_dimension == 1:
    """
        Calculate the PDF at slices of beta, and plot alongside the mode and stdev of the mode.
    """

    beta_slices = np.array([0, 0.25, 0.5, 0.75, 1]) * (beta_max - beta_min) + beta_min
    for beta in beta_slices:
        print("Starting PDF calculation for mass %.2E..." % beta)

        PDF = [func.dn(beta, rho) for rho in rho_vals]
        mode, mode_stdev = func.pdf_sample_expectation(PDF, rho_vals)

        # print("Sample mode: %.5E" % mode)
        # print("Sample variance from the mode: %.5E" % mode_stdev)
        # print("---------")

        plt.plot(rho_vals, PDF)
        plt.errorbar(mode, func.dn(beta, mode), xerr=mode_stdev, fmt='r.', markersize=10)
    
    plt.xlabel(r"$\tilde{\rho} (\rho/\bar{\rho}_m)$")
    plt.ylabel(r"$P_n$")
    plt.xscale("log")
    plt.title("PDF slices along mass")
    plt.grid(True)
    plt.legend(["%.2E" % beta for beta in beta_slices])

    plt.savefig("plots/joint-pdf-slice.pdf")
    plt.close()

"""
num_deltas = [25, 50, 100, 200]
if pms.run_convergence_tests:
    func.convergence_test(func.dn, func.pdf_analytic_expectation, func.pdf_sample_expectation, 
                            m_min, pms.beta_dd, delta_l_min, delta_l_max, num_deltas)
"""

print("Printed density profile.")