"""
    Filename: double_distribution_functions.py
    Author: Ericka Florio
    Created: 21 Oct 2025
    Description: Functions used solely by the double_distribution module, 
    including the full probability density and its analytic statistic estimates.
"""
import numpy as np
from numpy.typing import NDArray
import numpy.polynomial.polynomial as poly
from scipy.optimize import minimize
from scipy.special import erf
import scipy.integrate as integrate

import util.parameters as pms
import util.functions as func


def a_coll_integrand(x, c1, c2) -> float:
    return np.sqrt(x / (c1 * x**3 - c2 * x + 1))

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

def delta_c_0(a_i : float) -> float:
    """
        Calculate the critical overdensity today using a_coll and the growth 
        factor.
    """
    a_c = a_coll()
    delta_c = 3 * pms.Omega_m * (pms.kappa - pms.phi) * func.D(a_c) / 2
    temp = func.D(1) * delta_c / func.D(a_i)
    return temp

def CDF(rho, m:float = pms.M_200, beta:float = 1.3, a:float = 1):
    """
        Calculate the CDF as a function of delta_l(rho)
    """
    delta = rho - 1
    delta_c = delta_c_0(a) * func.D(a) / func.D(1)
    delta_tilde = delta_c * (1 - pow(1 + delta, -1/delta_c))
    rho_m = pms.Omega_m * pms.rho_c 

    N = (rho_m / m) * 1. / (2 * np.pi * np.sqrt(func.S(beta * m)) 
        * pow(func.S(m) - func.S(beta * m), 3/2))

    A = func.S(m) / (2 * func.S(beta * m) *(func.S(m) - func.S(beta * m)))
    B = delta_c_0(a) / (2 * (func.S(m) - func.S(beta * m)))
    C = (delta_c_0(a) ** 2) / (2 * (func.S(m) - func.S(beta * m)))

    cdf_temp = np.sqrt(np.pi / A) * (delta_c_0(a) - B / A) / 2. # 0.5 * np.sqrt(np.pi / A) * (delta_c_0(a) - B / (2 * A))
    cdf_temp *= np.exp(B**2 / A - C) * (1 - erf((B - A * delta_tilde) / np.sqrt(A))) # -1 np.sqrt(A) * delta_tilde - B / np.sqrt(A)
    cdf_temp += np.exp(-A * (delta_tilde**2) + 2 * B * delta_tilde - C) / (2 * A)
    cdf_temp *= N

    return cdf_temp

def conditional_CDF(rho, m:float = pms.M_200, beta:float = 1.3, a:float = 1):
    norm = (CDF(pms.rho_tilde_max, m, beta, a) 
            - CDF(pms.rho_tilde_min, m, beta, a))
    return CDF(rho, m, beta, a) / norm

def numeric_CDF(pdf, x_vals, x):
    i = np.argmin(np.abs(x_vals - x))
    return sum(pdf[:(i+1)])

def pdf_sample_expectation(pdf : NDArray, rho_vals : NDArray):
    """
        Calculate the mode of the double distribution 
        (i.e. the most probable profile)
        and the standard deviation of the mode, sliced at m.
    """

    sample_mode = rho_vals[np.argmax(pdf)]

    # sample_mode_variance = 0
    # for i in range(len(rho_vals)):
    #     sample_mode_variance += pdf[i] * pow(rho_vals[i] - sample_mode, 2)
    # sample_mode_variance /= (pms.num_rho - 1)

    return sample_mode #, np.sqrt(sample_mode_variance)

def analytic_IQR(sample_mode, sample_stdev, beta,
                 m:float = pms.M_200, a:float = 1) -> tuple[float, float]:
    """
        Find the analytic IQR by calculating CDF^-1(0.25), CDF^-1(0.75) via 
        root finding, then transform this value into rho from delta_tilde.
    """
    def find_IQR(q):
        if q == "l": 
            zscore = 0.25
            guess = sample_mode - sample_stdev
        elif q == "h":
            zscore = 0.75
            guess = sample_mode + sample_stdev
        else:
            print("Quantile specified cannot be computed.")
            exit()

        cdf_diff = lambda x: abs(conditional_CDF(x, m, beta, a) - zscore)
        soln = minimize(cdf_diff, guess, 
                        bounds=[(pms.rho_tilde_min, pms.rho_tilde_max)],
                        tol=pms.root_finder_precision)

        # print("-----")
        # print(sample_mode, sample_stdev)
        # print(guess, soln.x[0], soln.status)
        # print(conditional_CDF(pms.rho_tilde_max, m, beta, a), conditional_CDF(pms.rho_tilde_min, m, beta, a))
        # print(cdf_diff(soln.x[0]), cdf_diff(guess))
        # print(conditional_CDF(soln.x[0], m, beta, a))
        

        return soln.x[0]

    return find_IQR("l"), find_IQR("h")

def numeric_IQR(pdf, x_range):
    sm = 0
    iqrl, iqru = 0, 0
    cl, cu = 0, 0

    for idx, x in enumerate(x_range):
        sm += pdf[idx]
        if sm > 0.25 and cl == 0:
            iqrl = x
            cl += 1
        elif sm > 0.75 and cu == 0:
            iqru = x
            cu += 1
            break
    
    return iqrl, iqru

def dn(rho, m, beta, a:float = 1, transform_pdf:bool = pms.transform_pdf):
    """
        Calculate the double distribution of number density w/r/t mass and 
        local overdensity
    """

    delta_c = delta_c_0(a) * func.D(a) / func.D(1)
    rho_m = pms.Omega_m * pms.rho_c 

    # Transform rho into \tilde{\delta}_l
    delta = rho - 1
    delta_tilde = delta_c * (1 - pow(1 + delta, -1/delta_c))

    mass_removal = (delta_c_0(a) - delta_tilde) 
    mass_removal *= np.exp(-(delta_c_0(a) - delta_tilde)**2 
                            / (2 *(func.S(m) - func.S(beta*m)))) 
    mass_removal /= pow(func.S(m) - func.S(beta*m), 3/2)

    random_walk = (rho_m / m) * (np.exp(-(delta_tilde ** 2) / (2 * func.S(beta * m))) 
                    / (2 * np.pi * np.sqrt(func.S(beta*m))))

    # Construct the PDF
    dn = random_walk * mass_removal # * func.dS(m)

    # Apply Jacobian to get the transformed PDF
    if transform_pdf:
        dn *= pow(rho, (-1 - 1/delta_c))

    # Enforce the PDF to be positive
    if pms.enforce_positive_pdf == True:
        if dn < 0:
            return 0
        else:
            return dn
    else:
        return dn

def most_probable_rho(beta:float, gamma:float = pms.default_gamma, a:float = 1,
                        inc_mass_scaling:bool = False, m:float = pms.M_200):
    # From Eqn. 2 of arXiv:2404.11183v2 
    delta_c = delta_c_0(a) * func.D(a) / func.D(1)
    us_mode = pow(1 - pow(beta, -gamma), -delta_c + 1)
    
    if inc_mass_scaling:
        A = 1 / (func.S(m) - func.S(beta * m)) + 1 / (func.S(beta * m))
        B = - delta_c_0(a) * (2 / (func.S(m) - func.S(beta * m)) + 1 / func.S(beta * m))
        C = pow(delta_c_0(a), 2) / (func.S(m) - func.S(beta * m)) - 1

        roots = poly.polyroots([C, B, A])
        return roots[np.argmin(abs(roots - us_mode))]

    else:
        return us_mode

def most_probable_rho_transformed(m:float, beta:float, a:float = 1):
    """
        Calculate the expected rho from the correctly transformed double dist.
    """

    delta_c = delta_c_0(a) * func.D(a) / func.D(1)
    eta = delta_c_0(a) - delta_c
    A = func.S(m) / (2 * func.S(beta * m) *(func.S(m) - func.S(beta * m)))
    B = delta_c_0(a) / 2 / (func.S(m) - func.S(beta * m))

    Ap = A * pow(delta_c, 2)
    Bp = delta_c * (delta_c * A - B)

    a = 2 * Ap * delta_c
    b = 2 * (Ap * eta - Bp * delta_c)
    c = - (2 * Bp * eta + 2 * delta_c + pow(delta_c, 2))
    d = - eta * (1 + delta_c)

    roots = poly.polyroots([d, c, b, a])
    candidate = 0
    for i in range(len(roots)):
        if roots[i] > 0:
            candidate = pow(roots[i], -delta_c)

    return candidate