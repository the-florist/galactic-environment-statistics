"""
    Filename: collapse.py
    Author: Ericka Florio
    Description: Calculation of the collapse overdensity delta_c_0 (via the
                 collapse time a_coll), and the conversions between the rescaled
                 density rho and the linearly-extrapolated overdensity delta_tilde.
"""

import numpy as np
import scipy.integrate as integrate
from functools import lru_cache
from scipy.optimize import minimize

import gal_env_stats.parameters as pms
from gal_env_stats.physics.growth import D

def a_coll_integrand(x, c1, c2) -> float:
    return np.sqrt(x / (c1 * x**3 - c2 * x + 1))

@lru_cache(maxsize=None)
def a_coll() -> float:
    """
        Calculate a_coll by minimizing the integral to a_coll and the integral
        to a_pta
    """
    a_init = 1

    a_pta = pow(pms.w, -1/3)
    a_pta *= np.sqrt(4 * pms.kappa / 3 / pow(pms.w, 1/3))
    a_pta *= np.cos(1/3 * (np.arccos(np.sqrt(27 *
                pow(pms.kappa/pow(pms.w, 1/3), -3) / 4)) + np.pi))

    C = 2 * integrate.quad(lambda x: a_coll_integrand(x, pms.w, pms.kappa),
                            pms.a_i, a_pta)[0]

    def diff(a):
        return abs(integrate.quad(lambda x: a_coll_integrand(x, pms.w, pms.phi),
                                    pms.a_i, a)[0] - C)

    solution = minimize(diff, a_init, bounds=[(0, 1)], tol=pms.root_finder_precision)

    return solution.x[0]

@lru_cache(maxsize=None)
def delta_c_0(a_i : float) -> float:
    """
        Calculate the critical overdensity today using a_coll and the growth
        factor.
    """
    a_c = a_coll()
    delta_c = 3 * pms.Omega_m * (pms.kappa - pms.phi) * D(a_c) / 2
    temp = D(1) * delta_c / D(a_i)
    return temp

def rho_to_delta_tilde(rho, a:float = 1):
    delta_c = delta_c_0(a) * D(a) / D(1)
    return delta_c * (1 - pow(rho, -1/delta_c))

def delta_tilde_to_rho(delta_tilde:float, a:float = 1):
    delta_c = delta_c_0(a) * D(a) / D(1)
    return pow(1 - delta_tilde / delta_c, -delta_c)
