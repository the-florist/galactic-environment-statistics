"""
    Author: Ericka Florio
    Created: 8th September 2025
    Description: Shared functions for calculating the most probable density profile.
"""

# libraries
import scipy.integrate as integrate
import numpy as np

# parameters
import parameters as mp

def D_integrand(x, Om_integrand : float, Ol_integrand : float):
    return pow(x / (x * (1 - Om_integrand - Ol_integrand) + Om_integrand + Ol_integrand * (x ** 3)), 3/2)

def D(a, floor, return_full = False, Om : float = mp.Omega_m, Ol : float = mp.Omega_L):
    out_full = integrate.quad(lambda x: D_integrand(x, Om, Ol), floor, a)
    D_temp = out_full[0]
    D_temp *= np.sqrt(a * (1 - Om - Ol) + Om + Ol * (a ** 3)) / pow(a, 3/2)
    err = out_full[1]
    
    if(return_full):
        return [D_temp, err]
    else:
        return D_temp

x_of_a = lambda a: pow(2 * mp.w, 1/3) * a
A_integrand = lambda u: pow(u / (pow(u, 3) + 2), 3/2)

def A(x_val):
        out = integrate.quad(A_integrand, x_of_a(mp.a_i), x_val)
        A_tmp = out[0]
        A_tmp *= np.sqrt(pow(x_val, 3) + 2) / pow(x_val, 3/2)
        return A_tmp


def rho(beta, gamma, delta_c, Omega_m, a = 1):
    return (Omega_m * mp.rho_c) * (a ** -3) * pow(1 - pow(beta, -gamma), -delta_c + 1)

def r(beta, gamma, delta_c, w):
    return pow(w * beta * pow(1 - pow(beta, -gamma), delta_c), 1/3)