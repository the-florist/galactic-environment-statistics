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

Omega_m = 0.3
Omega_L = 0.7
delta_c = 1.6757
gamma = 0.52
w = Omega_L/Omega_m

beta_0 = 1.3
beta_f = 3
num_betas = 300

print("Cosmological parameters:")
print("Omega_m = "+str(Omega_m))
print("Omega_L = "+str(Omega_L))
print("gamma = "+str(gamma))
print("--------")

beta_range = np.linspace(beta_0, beta_f, num_betas)
rhos = [func.rho(b, gamma, delta_c, Omega_m) for b in beta_range]
rs = [func.r(b, gamma, delta_c, w) for b in beta_range]

plt.plot(rs, rhos, label="Density profile")
plt.xlabel('r (R/R_ta)')
plt.ylabel("rho [M_s/Mpc^3]")
plt.title("Mock density profile (today, LambdaCDM, single gamma)")
plt.legend()
plt.grid(True)
plt.savefig("density-profile.pdf")
plt.close()

print("Printed density profile.")