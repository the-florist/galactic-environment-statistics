"""
    Filename: double-distribution.py
    Author: Ericka Florio
    Created: 11 Sept 2025
    Description: Plots the joint double distribution for the number density of objects 
            of mass m with local overdensity delta_l, 
            as derived in Pavlidou and Fields 2005.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import parameters as pms 
import functions as func

rho_m_0 = pms.Omega_m * pms.rho_c

def dn(m, delta_l, beta, delta_c_0):
    mass_removal = (delta_c_0 - delta_l) * np.exp(-(delta_c_0 - delta_l)**2 / (2 *(func.S(m) - func.S(beta*m)))) 
    mass_removal /= pow(func.S(m) - func.S(beta*m), 3/2)

    random_walk = np.exp(-(delta_l**2) / (2 * func.S(beta*m))) / (2 * np.pi * np.sqrt(func.S(beta*m))) # rho_m_0 / m 
    return mass_removal * random_walk

beta = 1.7
delta_c_0 = 0.4

# Define ranges for m and delta_l
m_min, m_max = 1e14, 6e15
delta_l_min, delta_l_max = -2, 2
num_m = 100
num_delta_l = 100

print("Overall scale:")
print(rho_m_0/m_min, rho_m_0/m_max)

m_vals = np.logspace(np.log10(m_min), np.log10(m_max), num_m)
delta_l_vals = np.linspace(delta_l_min, delta_l_max, num_delta_l)

# Create meshgrid for m and delta_l
M, DL = np.meshgrid(m_vals, delta_l_vals, indexing='ij')

# Compute joint pdf using dn
Z = np.zeros_like(M)
for i in range(num_m):
    for j in range(num_delta_l):
        try:
            Z[i, j] = dn(M[i, j], DL[i, j], beta, delta_c_0)
        except Exception:
            Z[i, j] = 0

# Marginals
marginal_m = np.trapezoid(Z, delta_l_vals, axis=1)
marginal_dl = np.trapezoid(Z, m_vals, axis=0)

# Set up the figure with joint and marginal axes
from matplotlib.gridspec import GridSpec

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
plt.savefig("joint_pdf_m_delta_l.png", dpi=150)
plt.close()

print("Printed density profile.")