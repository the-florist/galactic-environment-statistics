"""
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

# Information to terminal
print("Cosmological parameters:")
print("Omega_m = "+str(pms.Omega_m))
print("Omega_L = "+str(pms.Omega_L))
print("gamma = "+str(gamma))
print("delta_c = "+str(delta_c))
print("--------")

# Scaling r with delta_ta
w = lambda g:(1 + delta_ta) * pow(1 - pow(1 + delta_ta, -1/delta_c), 1/g)

# Solve for rho and r parametrically 
beta_range = np.linspace(beta_0, beta_f, num_betas)
rhos = [[func.rho(b, g, delta_c, pms.Omega_m) for b in beta_range] for g in gamma]
rs = [[func.r(b, g, delta_c, w(g)) for b in beta_range] for g in gamma]

# Plot this
for i in np.arange(0, len(gamma)):
    plt.plot(rs[i], rhos[i], label=r"$\gamma =$ "+str(gamma[i]))
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