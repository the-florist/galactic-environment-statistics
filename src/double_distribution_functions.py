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

import util.parameters as pms
import util.functions as func
from util.functions import delta_c_0

"""
    Functions related to the double distribution (PDF).
"""

def dn(rho, m, beta, a:float = 1, transform_pdf:bool = pms.transform_pdf):
    """
        Calculate the double distribution of number density w/r/t mass and 
        local overdensity
    """

    # Set some useful constants
    delta_c = delta_c_0(a) * func.D(a) / func.D(1)
    rho_m = pms.Omega_m * pms.rho_c 

    # Transform rho into \tilde{\delta}_l
    delta_tilde = func.rho_to_delta_tilde(rho)

    # Calculate the component distributions
    mass_removal = (delta_c_0(a) - delta_tilde) 
    mass_removal *= np.exp(-(delta_c_0(a) - delta_tilde)**2 
                            / (2 *(func.S(m) - func.S(beta*m)))) 
    mass_removal /= pow(func.S(m) - func.S(beta*m), 3/2)

    random_walk = (rho_m / m) * (np.exp(-(delta_tilde ** 2) / (2 * func.S(beta * m))) 
                    / (2 * np.pi * np.sqrt(func.S(beta*m))))

    # Construct the raw PDF
    dn = random_walk * mass_removal # * func.dS(m)

    # Apply inverse derivative to get the transformed PDF of rho
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
    """
        Find the most probable rho under the most restrictive assumptions,
        either with or without mass dependence.
    """

    delta_c = delta_c_0(a) * func.D(a) / func.D(1)
    # From Eqn. 2 of arXiv:2404.11183v2 
    us_mode_rho = pow(1 - pow(beta, -gamma), -delta_c + 1)
    # From Eqn. 3 of arXiv:2402.18634v2
    us_mode_delta = delta_c_0(a) * func.S(beta * m) / func.S(m)
    
    if inc_mass_scaling:
        # Solve the quadratic, which keeps the dependence of the mode on mass.
        A = 1 / (func.S(m) - func.S(beta * m)) + 1 / (func.S(beta * m))
        B = - delta_c_0(a) * (2 / (func.S(m) - func.S(beta * m)) + 1 / func.S(beta * m))
        C = pow(delta_c_0(a), 2) / (func.S(m) - func.S(beta * m)) - 1

        candidates = poly.polyroots([C, B, A])
        root = candidates[np.argmin(abs(candidates - us_mode_delta))]
        return func.delta_tilde_to_rho(root)

    else:
        # Return the universal profile, which does not depend on mass.
        return us_mode_rho

def most_probable_rho_transformed(m:float, beta:float, a:float = 1):
    """
        Calculate the most probable rho, correctly transforming delta_tilde -> rho.
        Involves solving the roots of a third-order polynomial.
    """

    # Set up first layer of constants
    delta_c = delta_c_0(a) * func.D(a) / func.D(1)
    eta = delta_c_0(a) - delta_c
    A = func.S(m) / (2 * func.S(beta * m) *(func.S(m) - func.S(beta * m)))
    B = delta_c_0(a) / 2 / (func.S(m) - func.S(beta * m))

    # Set up second layer of constants
    Ap = A * pow(delta_c, 2)
    Bp = delta_c * (delta_c * A - B)

    # Set up third layer of constants
    a = 2 * Ap * delta_c
    b = 2 * (Ap * eta - Bp * delta_c)
    c = - (2 * Bp * eta + 2 * delta_c + pow(delta_c, 2))
    d = - eta * (1 + delta_c)

    # Solve the cubic
    roots = poly.polyroots([d, c, b, a])
    for i in range(len(roots)):
        if roots[i] > 0:
            return pow(roots[i], -delta_c)
    else:
        print("Error : most_probable_rho_transformed, Root not found.")
        exit()

"""
    Functions related to the CDF.
"""

def CDF(rho, m:float = pms.M_200, beta:float = 1.3, a:float = 1):
    """
        Calculate the analytic CDF as a function of delta_l(rho).
    """

    # Convert rho -> delta_tilde and set constants.
    delta_tilde = func.rho_to_delta_tilde(rho)
    rho_m = pms.Omega_m * pms.rho_c 

    N = (rho_m / m) * 1. / (2 * np.pi * np.sqrt(func.S(beta * m)) 
        * pow(func.S(m) - func.S(beta * m), 3/2))

    A = func.S(m) / (2 * func.S(beta * m) *(func.S(m) - func.S(beta * m)))
    B = delta_c_0(a) / (2 * (func.S(m) - func.S(beta * m)))
    C = (delta_c_0(a) ** 2) / (2 * (func.S(m) - func.S(beta * m)))

    # Calculate the CDF in layers.
    cdf_temp = np.sqrt(np.pi / A) * (delta_c_0(a) - B / A) / 2. # 0.5 * np.sqrt(np.pi / A) * (delta_c_0(a) - B / (2 * A))
    cdf_temp *= np.exp(B**2 / A - C) * (1 - erf((B - A * delta_tilde) / np.sqrt(A))) # -1 np.sqrt(A) * delta_tilde - B / np.sqrt(A)
    cdf_temp += np.exp(-A * (delta_tilde**2) + 2 * B * delta_tilde - C) / (2 * A)
    cdf_temp *= N

    return cdf_temp

def conditional_CDF(rho, m, beta, a:float = 1):
    """
        Calculate the normalised analytic CDF as a function of delta_l(rho).
    """

    norm = (CDF(pms.rho_tilde_max, m, beta, a) 
            - CDF(pms.rho_tilde_min, m, beta, a))
    return CDF(rho, m, beta, a) / norm

def numeric_CDF(pdf, x_vals, x):
    """
        Find the numerical CDF given a numerical PDF and axis.
    """

    i = np.argmin(np.abs(x_vals - x))
    return sum(pdf[:(i+1)])

"""
    Functions related to the IQR.
"""

def pdf_sample_expectation(pdf : NDArray, rho_vals : NDArray):
    """
        Calculate the mode of the double distribution 
        (i.e. the most probable profile)
        and the standard deviation of the mode, sliced at m.
    """

    sample_mode = rho_vals[np.argmax(pdf)]

    sample_mode_variance = 0
    for i in range(len(rho_vals)):
        sample_mode_variance += pdf[i] * pow(rho_vals[i] - sample_mode, 2)
    sample_mode_variance /= (pms.num_rho - 1)

    return sample_mode, np.sqrt(sample_mode_variance)

def analytic_IQR(sample_mode, sample_stdev, beta, mass, 
                 a:float = 1) -> tuple[float, float]:
    """
        Find the analytic IQR by calculating CDF^-1(0.25), CDF^-1(0.75) via 
        root finding, then transform this value into rho from delta_tilde.
    """
    def find_IQR(q):
        if q == "l": 
            zscore = pms.lqr
            guess = sample_mode - sample_stdev
        elif q == "h":
            zscore = pms.uqr
            guess = sample_mode + sample_stdev
        else:
            print("Quantile specified cannot be computed.")
            exit()

        cdf_diff = lambda x: abs(conditional_CDF(x, mass, beta, a) - zscore)
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
    """
        Calculate the IQR numerically, on a numeric PDF with axis.
    """

    # Track the CDF, and iqrs found
    sm = 0
    iqrl, iqru = 0, 0
    cl, cu = 0, 0

    # Use the numerical CDF to find the iqrs
    for idx, x in enumerate(x_range):
        sm += pdf[idx]
        if sm > pms.lqr and cl == 0:
            iqrl = x
            cl += 1
        elif sm > pms.uqr and cu == 0:
            iqru = x
            cu += 1
            break
    
    return iqrl, iqru