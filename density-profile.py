"""
    Author: Ericka Florio
    Date: 8th September 2025
    Description: Calculates the parametric density profile (rho(beta), R(beta))
    given values for the collapse density contrast, the variance power law exponent, 
    and the cosmology
"""

import numpy as np
import matplotlib.pyplot as plt

# Enable mathtext rendering (doesn't require LaTeX installation)
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'

import parameters as pms 
import functions as func

print("Visualising density profile.")

# Problem-specific quantities
delta_c = 1.6757 # requires flat LambdaCDM universe
gamma = 0.52     # taken as mean of figure given in Pavlidou 2024
delta_ta = 11 * (pms.Omega_m)

# Integration range
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

w = (1 + delta_ta) * pow(1 - pow(1 + delta_ta, -1/delta_c), 1/gamma)

beta_range = np.linspace(beta_0, beta_f, num_betas)
rhos = [func.rho(b, gamma, delta_c, pms.Omega_m) for b in beta_range]
rs = [func.r(b, gamma, delta_c, w) for b in beta_range]

plt.plot(rs, rhos, label="Density profile")
plt.xlabel(r"$r$ ($R/R_{\mathrm{ta}}$)")
plt.xscale('log')
plt.ylabel(r"$\tilde{\rho}$ $(\rho/\rho_{m})$")
plt.yscale('log')
plt.title(r"Mock density profile (today, LambdaCDM, $\gamma=0.52$)")
plt.legend()
plt.grid(True)
plt.savefig("rho-vs-r.pdf")
plt.close()

print("Figure printed.")