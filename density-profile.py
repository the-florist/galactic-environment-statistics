"""
    Filename: density-profile.py
    Author: Ericka Florio
    Date: 8th September 2025
    Description: Calculates the parametric density profile (rho(beta), R(beta))
    given values for the collapse density contrast, the variance power law exponent, 
    and the cosmology
"""

import numpy as np
import matplotlib.pyplot as plt

import parameters as pms 
import functions as func

print("Visualising density profile.")

# Problem-specific quantities
delta_c = 1.6757                # requires flat LambdaCDM universe
gamma = [0.55, 0.525, 0.50]     # taken as mean of figure given in Pavlidou 2024
delta_ta = 11 * (pms.Omega_m)   # turnaround overdensity (provided by Vaso)

# Integration range 
# turnaround predicted around beta ~ 1.7
# the model cannot work below beta = 1.2
beta_0 = 1.3
beta_f = 10
num_betas = 300
beta_range = np.linspace(beta_0, beta_f, num_betas)

# Information to terminal
print("Cosmological parameters:")
print("Omega_m = "+str(pms.Omega_m))
print("Omega_L = "+str(pms.Omega_L))
print("delta_c = "+str(delta_c))

if pms.power_law_approx == True:
    print("Using power law approximation for S(m).\n gamma = "+str(gamma))
    print("--------")

    # Solve for rho and r parametrically 
    rhos = [[func.rho(b, delta_c, gamma = g) for b in beta_range] for g in gamma]
    rs = [[func.r(b, delta_c, delta_ta, gamma = g) for b in beta_range] for g in gamma]

    # Plot this
    for i in np.arange(0, len(gamma)):
        plt.plot(rs[i], rhos[i], label=r"$\gamma =$ "+str(gamma[i]))

else:
    print("Using Bardeen transfer function for S(m)")
    masses = [2e13, 4e13, 6e13, 8e13, 1e14]  # Solar masses

    rhos = [[func.rho(beta, delta_c, m = mass) for beta in beta_range] for mass in masses]
    rs = [[func.r(beta, delta_c, delta_ta, m = mass) for beta in beta_range] for mass in masses]

    for m in np.arange(0, len(masses)):
        plt.plot(rs[m], rhos[m], label=rf"$m = {masses[m]:.1e}$")

plt.xlabel(r"$r$ ($R/R_{\mathrm{ta}}$)")
plt.xscale('log')
plt.ylabel(r"$\tilde{\rho}$ $(\rho/\rho_{m})$")
plt.yscale('log')
plt.title(r"Mock density profile (today, LambdaCDM)")
plt.legend()
plt.grid(True)
plt.savefig("rho-vs-r.pdf")
plt.close()



print("Figure printed.")