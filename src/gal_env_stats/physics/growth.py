"""
    Filename: growth.py
    Author: Ericka Florio
    Description: Calculation of the linear growth factor D(a), and the A(x)
                 helper used for the free-LambdaCDM analytic check.
"""

import numpy as np
import scipy.integrate as integrate
from typing import overload, Literal, Tuple, Union

import gal_env_stats.parameters as pms

x_of_a = lambda a: pow(2 * pms.w, 1/3) * a
A_integrand = lambda u: pow(u / (pow(u, 3) + 2), 3/2)

def A(x_val):
        out = integrate.quad(A_integrand, x_of_a(pms.a_i), x_val)
        A_tmp = out[0]
        A_tmp *= np.sqrt(pow(x_val, 3) + 2) / pow(x_val, 3/2)
        return A_tmp

def D_integrand(x: float, Om_integrand: float, Ol_integrand: float) -> float:
    return pow(x / (x * (1 - Om_integrand - Ol_integrand) + Om_integrand
                + Ol_integrand * (x ** 3)), 3/2)

@overload
def D(a: float, return_full: Literal[True], Om: float = ..., Ol: float = ...) -> Tuple[float, float]:
    ...

@overload
def D(a: float, return_full: Literal[False] = ..., Om: float = ..., Ol: float = ...) -> float:
    ...

def D(a: float, return_full: bool = False, Om: float = pms.Omega_m, Ol: float = pms.Omega_L) -> Union[float, Tuple[float, float]]:
    out_full = integrate.quad(lambda x: D_integrand(x, Om, Ol), pms.a_i, a)
    D_temp = out_full[0]
    D_temp *= np.sqrt(a * (1 - Om - Ol) + Om + Ol * (a ** 3)) / pow(a, 3/2)
    err = out_full[1]

    if(return_full):
        return (D_temp, err)
    else:
        return D_temp
