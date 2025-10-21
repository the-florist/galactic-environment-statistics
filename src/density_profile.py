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
import scipy.integrate as integrate

import util.parameters as pms 
import util.functions as func

# Problem-specific quantities
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
print("delta_c = "+str(pms.delta_c))

def rho_avg(Sbm, Sm, delta_c):
    return (pms.Omega_m * pms.rho_c) * pow(1 - Sbm/Sm, -delta_c)

def rho(beta, delta_c, gamma:float = pms.default_gamma, a = 1, m:float = pms.M_200):
    """
        Find rho(beta) for power law approximation of S(m),
        or rho(beta, m) for transfer function version of S(m).
    """
    if pms.power_law_approx == True:
        return ((pms.Omega_m * pms.rho_c) * (a ** -3) 
                * pow(1 - pow(beta, -gamma), -delta_c + 1) 
                / (pms.Omega_m * pms.rho_c))

    else:
        C = pms.s_8 / (integrate.quad(lambda k: func.transfer_function_integrand(k), 
                                        0, func.k_of_m(pms.m_8)))[0]
        temp = delta_c * pow(1 - func.S(beta * m)/func.S(m), -1) * func.k_of_m(beta * m) 
        temp *= func.transfer_function_integrand(func.k_of_m(m)) / 3 / func.S(m)
        denominator = 1 - temp * C
        return rho_avg(func.S(beta * m), func.S(m), delta_c) / denominator

def r(beta, delta_c, delta_ta, gamma:float = pms.default_gamma, m:float = pms.M_200):
    """
        Find r(beta) for power law approximation of S(m), where r = R/R_ta,
        or r(beta, m) for transfer function version of S(m).
    """
    if pms.power_law_approx == True:
            w = (1 + delta_ta) * pow(1 - pow(1 + delta_ta, -1/delta_c), 1/gamma)
            return pow(w * beta * pow(1 - pow(beta, -gamma), delta_c), 1/3)

    else:
        temp = beta * pow(1 - func.S(beta * m)/func.S(m), delta_c) * (1 + delta_ta) / pms.beta_ta # FIXME
        return pow(temp, 1/3)


def run():
    """
        Calculate and plot the density profile for given model of S(m)
    """
    if pms.power_law_approx == True:
        print("Using power law approximation for S(m).\n gamma = "+str(gamma))

        # Solve for rho and r parametrically 
        rhos = [[rho(b, pms.delta_c, gamma = g) for b in beta_range] 
                 for g in gamma]
        rs = [[r(b, pms.delta_c, delta_ta, gamma = g) for b in beta_range] 
               for g in gamma]

        # Plot this
        for i in np.arange(0, len(gamma)):
            plt.plot(rs[i], rhos[i], label=r"$\gamma =$ "+str(gamma[i]))

    else:
        print("Using Bardeen transfer function for S(m)")
        masses = [2e13, 4e13, 6e13, 8e13, 1e14]  # Solar masses

        rhos = [[rho(beta, pms.delta_c, m = mass) 
                 for beta in beta_range] 
                 for mass in masses]
        rs = [[r(beta, pms.delta_c, delta_ta, m = mass) 
               for beta in beta_range] 
               for mass in masses]

        for m in np.arange(0, len(masses)):
            plt.plot(rs[m], rhos[m], label=rf"$m = {masses[m]:.1e}$")

    plt.xlabel(r"$r$ ($R/R_{\mathrm{ta}}$)")
    plt.xscale('log')
    plt.ylabel(r"$\tilde{\rho}$ $(\rho/\rho_{m})$")
    plt.yscale('log')
    plt.title(r"Mock density profile (today, LambdaCDM)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/density-profile.pdf")
    plt.close()