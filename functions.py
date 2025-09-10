"""
    Author: Ericka Florio
    Created: 8th September 2025
    Description: Shared functions for calculating the most probable density profile.
"""

# libraries
import scipy.integrate as integrate
import numpy as np

# parameters
import parameters as pms


"""
    Functions used in growth-factor, to calculate D(a)
"""

x_of_a = lambda a: pow(2 * pms.w, 1/3) * a
A_integrand = lambda u: pow(u / (pow(u, 3) + 2), 3/2)

def A(x_val):
        out = integrate.quad(A_integrand, x_of_a(pms.a_i), x_val)
        A_tmp = out[0]
        A_tmp *= np.sqrt(pow(x_val, 3) + 2) / pow(x_val, 3/2)
        return A_tmp

def D_integrand(x, Om_integrand : float, Ol_integrand : float):
    return pow(x / (x * (1 - Om_integrand - Ol_integrand) + Om_integrand + Ol_integrand * (x ** 3)), 3/2)

def D(a, floor, return_full = False, Om : float = pms.Omega_m, Ol : float = pms.Omega_L):
    out_full = integrate.quad(lambda x: D_integrand(x, Om, Ol), floor, a)
    D_temp = out_full[0]
    D_temp *= np.sqrt(a * (1 - Om - Ol) + Om + Ol * (a ** 3)) / pow(a, 3/2)
    err = out_full[1]
    
    if(return_full):
        return [D_temp, err]
    else:
        return D_temp


"""
    Functions used in density-profile.py to calculate (r, rho)
"""

k_of_m = lambda m: pow(6 * (np.pi ** 2) * (pms.Omega_m * pms.rho_c) / m, 1/3)
q_of_k = lambda k: k / pms.Omega_m / pow(pms.h, 2)

def rho_avg(Sbm, Sm, delta_c):
    return (pms.Omega_m * pms.rho_c) * pow(1 - Sbm/Sm, -delta_c)

def transfer_function_integrand(k):
    """
        The integrand used to calculate S(m) according to Bardeen's transfer function.
    """
    transfer_function = np.log(1 + 2.34 * q_of_k(k)) 
    transfer_function *= pow(1 + 3.89 * q_of_k(k) + pow(16.1 * q_of_k(k), 2) + pow(5.46 * q_of_k(k), 3) + pow(6.71 * q_of_k(k), 4), -1/4)
    transfer_function /= (2.34 * q_of_k(k))

    temp = pow(transfer_function, 2) * pow(k, 2 + pms.n)
    return temp


def S(m, gamma = 1, power_law_approx = pms.power_law_approx):
    """
        Variance of the density field 
        calculated both in the power law approximation
        and by the transfer function from Bardeen 1986.
    """
    if power_law_approx == True:
        return m ** (-gamma)

    else:
        S_temp = integrate.quad(lambda k: transfer_function_integrand(k), 0, k_of_m(m))
        S_temp *= pms.s_8
        S_temp /= (integrate.quad(lambda k: transfer_function_integrand(k), 0, k_of_m(pms.m_8)))
        return S_temp 

def rho(beta, gamma, delta_c, Omega_m, a = 1, power_law_approx = pms.power_law_approx, m = 1):
    """
        Find rho(beta) for power law approximation of S(m),
        or rho(beta, m) for transfer function version of S(m).
    """
    if power_law_approx == True:
        return (Omega_m * pms.rho_c) * (a ** -3) * pow(1 - pow(beta, -gamma), -delta_c + 1) / (Omega_m * pms.rho_c)

    else:
        C = pms.s_8 / (integrate.quad(lambda k: transfer_function_integrand(k), 0, k_of_m(pms.m_8)))
        temp = delta_c * pow(1 - S(beta * m)/S(m), -1) * k_of_m(beta * m) * transfer_function_integrand(k_of_m(m)) / 3 / S(m)
        denominator = 1 - temp * C
        return rho_avg(S(beta * m), S(m), delta_c) / denominator
        

def r(beta, gamma, delta_c, delta_ta, power_law_approx = pms.power_law_approx, m = 1):
    """
        Find r(beta) for power law approximation of S(m), where r = R/R_ta,
        or r(beta, m) for transfer function version of S(m).
    """
    if power_law_approx == True:
            w = (1 + delta_ta) * pow(1 - pow(1 + delta_ta, -1/delta_c), 1/gamma)
            return pow(w * beta * pow(1 - pow(beta, -gamma), delta_c), 1/3)

    else:
        temp = beta * pow(1 - S(beta * m)/S(m), delta_c) * (1 + delta_ta) / pms.beta_ta
        return pow(temp, 1/3)

