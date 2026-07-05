"""
    Filename: variance.py
    Author: Ericka Florio
    Description: Calculation of the matter variance S(m), both in the power-law
                 approximation and via Bardeen's adiabatic-CDM transfer function.
"""

import numpy as np
import scipy.integrate as integrate

import gal_env_stats.parameters as pms

k_of_m = lambda m: pow(6 * (np.pi ** 2) * (pms.Omega_m * pms.rho_c) / m, 1/3)
q_of_k = lambda k: k / pms.Omega_m / pow(pms.h, 2)

def transfer_function_integrand(k):
    """
        The integrand used to calculate S(m) according to Bardeen's
        transfer function.
    """
    transfer_function = np.log(1 + 2.34 * q_of_k(k))
    transfer_function *= pow(1 + 3.89 * q_of_k(k)
                        + pow(16.1 * q_of_k(k), 2)
                        + pow(5.46 * q_of_k(k), 3)
                        + pow(6.71 * q_of_k(k), 4), -1/4)

    transfer_function /= (2.34 * q_of_k(k))

    temp = pow(transfer_function, 2) * pow(k, 2 + pms.n)
    return temp


def S(m, gamma, pla = pms.power_law_approx):
    """
        Variance of the density field
        calculated both in the power law approximation
        and by the transfer function from Bardeen 1986.
    """
    if pla == True:
        return pms.s_8 * (m/pms.m_8) ** (-gamma)

    else:
        ms = np.unique(m)
        integral = {mv: integrate.quad(lambda k: transfer_function_integrand(k),
                    0, k_of_m(mv))[0] for mv in ms}
        S_temp = np.vectorize(lambda mv: integral[mv])(m)

        S_temp *= pms.s_8
        S_temp /= (integrate.quad(lambda k: transfer_function_integrand(k), 0, k_of_m(pms.m_8))[0])
        return S_temp

def dS(m, power_law_approx = pms.power_law_approx, gamma:float = pms.default_gamma):
    if power_law_approx:
        dS = - pms.s_8 * gamma * pow(m / pms.m_8, - gamma - 1) / pms.m_8
        return dS
    else:
        raise NotImplementedError(
            "Numerical derivative of S(m) is not yet implemented.")
