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

# Define ranges for m and delta_l (values from P&F 2005 Fig. 1)
m_min, m_max = 2e13, 1e14 # Solar masses
delta_l_min, delta_l_max = -1, 5
num_m = 100
num_delta_l = 100

m_vals = np.logspace(np.log10(m_min), np.log10(m_max), num_m)
delta_l_vals = np.linspace(delta_l_min, delta_l_max, num_delta_l)

print("-----\nDomain: ")
print(rf"Mass range: ({m_min:.1e}, {m_max:.1e})")
print("Density contrast range: ("+str(delta_l_min)+", "+str(delta_l_max)+")")

if pms.plot_dimension == 2:
    # Create meshgrid for m and delta_l
    M, DL = np.meshgrid(m_vals, delta_l_vals, indexing='ij')

    print("Starting loop over mass/contrast domain...")

    # Compute joint pdf using dn
    Z = np.zeros_like(M)
    for i in range(num_m):
        for j in range(num_delta_l):
            try:
                Z[i, j] = func.dn(M[i, j], DL[i, j], pms.beta_dd)
            except Exception:
                Z[i, j] = 0

    # Marginals
    marginal_m = np.trapezoid(Z, delta_l_vals, axis=1)
    marginal_dl = np.trapezoid(Z, m_vals, axis=0)

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)

    ax_joint = fig.add_subplot(gs[1:,:3])
    ax_marg_m = fig.add_subplot(gs[0,:3], sharex=ax_joint)
    ax_marg_dl = fig.add_subplot(gs[1:,3], sharey=ax_joint)

    # Joint pdf heatmap
    c = ax_joint.pcolormesh(m_vals, delta_l_vals, Z.T, shading='auto', cmap='viridis')
    ax_joint.set_xlabel('m')
    ax_joint.set_ylabel(r'$\delta_l$')
    ax_joint.set_xscale('log')
    # fig.colorbar(c, ax=ax_joint, label='dn(m, delta_l)')

    # Marginal for m
    ax_marg_m.plot(m_vals, marginal_m, color='tab:blue')
    ax_marg_m.set_xscale('log')
    ax_marg_m.set_ylabel('Marginal\nPDF (m)')
    ax_marg_m.tick_params(axis='x', labelbottom=False)
    ax_marg_m.grid(True, which='both', ls='--', alpha=0.3)

    # Marginal for delta_l
    ax_marg_dl.plot(marginal_dl, delta_l_vals, color='tab:orange')
    ax_marg_dl.set_xlabel('Marginal\nPDF ($\delta_l$)')
    ax_marg_dl.tick_params(axis='y', labelleft=False)
    ax_marg_dl.grid(True, which='both', ls='--', alpha=0.3)

    # Remove tick labels on shared axes
    plt.setp(ax_marg_m.get_xticklabels(), visible=False)
    plt.setp(ax_marg_dl.get_yticklabels(), visible=False)

    ax_joint.set_title('Joint PDF of m and $\delta_l$ with marginals')
    # plt.tight_layout()
    plt.savefig("plots/joint-probability.pdf")
    plt.close()

elif pms.plot_dimension == 1:
    mass_slices = np.array([0, 0.25, 0.5, 0.75, 1])
    mass_slices *= (m_max - m_min) 
    mass_slices += m_min

    for m in mass_slices:
        print("Starting PDF calculation for mass %.2E..." % m)
        plt.plot(np.array([func.dn(m, delta, pms.beta_dd) for delta in delta_l_vals]))
    
    plt.xlabel(r"$\delta_l$")
    plt.ylabel(r"$P_n$")
    plt.legend(["%.2E" % m for m in mass_slices])
    plt.title("PDF slices along mass")
    plt.savefig("plots/contrast-pdf-slice.pdf")
    plt.grid(True)
    plt.close()

print("Printed density profile.")