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
from scipy.special import erf

import util.parameters as pms
import util.functions as func
from util.functions import delta_c_0, delta_tilde_to_rho

"""
    Functions related to the double distribution (PDF).
"""

def dn(rho, m, beta, a:float = 1, transform_pdf:bool = pms.transform_pdf, 
                                  gamma:float = pms.default_gamma):
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
                            / (2 *(func.S(m, gamma) - func.S(beta * m, gamma)))) 
    mass_removal /= pow(func.S(m, gamma) - func.S(beta*m, gamma), 3/2)

    random_walk = (rho_m / m) * (np.exp(-(delta_tilde ** 2) / (2 * func.S(beta * m, gamma))) 
                    / (2 * np.pi * np.sqrt(func.S(beta * m, gamma))))

    # Construct the raw PDF
    dn = random_walk * mass_removal # * func.dS(m)

    # Apply inverse derivative to get the transformed PDF of rho
    if transform_pdf:
        dn *= pow(rho, (-1 - 1/delta_c))

    if pms.enforce_positive_pdf == True:
        dn = np.maximum(dn, 0)

    return dn

def most_probable_rho(m, beta, gamma:float = pms.default_gamma, a:float = 1,
                        inc_mass_scaling:bool = False):
    """
        Find the most probable rho under the most restrictive assumptions,
        either with or without mass dependence.
    """

    delta_c = delta_c_0(a) * func.D(a) / func.D(1)
    # From Eqn. 2 of arXiv:2404.11183v2 
    us_mode_rho = pow(1 - pow(beta, -gamma), -delta_c + 1)
    # From Eqn. 3 of arXiv:2402.18634v2
    us_mode_delta = delta_c_0(a) * func.S(beta * m, gamma) / func.S(m, gamma)
    
    if inc_mass_scaling:
        # Solve the quadratic, which keeps the dependence of the mode on mass.
        A = 1 / (func.S(m, gamma) - func.S(beta * m, gamma)) + 1 / (func.S(beta * m, gamma))
        B = - delta_c_0(a) * (2 / (func.S(m, gamma) - func.S(beta * m, gamma)) + 1 / func.S(beta * m, gamma))
        C = pow(delta_c_0(a), 2) / (func.S(m, gamma) - func.S(beta * m, gamma)) - 1

        try:
            roots = np.zeros_like(beta)
            for i in range(len(beta)):
                for j in range(len(beta[0])):
                    candidates = poly.polyroots([C[i, j], B[i, j], A[i, j]])
                    root = candidates[np.argmin(abs(candidates - us_mode_delta[i, j]))]
                    roots[i, j] = delta_tilde_to_rho(root)
            return roots
        
        except:
            candidates = poly.polyroots([C, B, A])
            root = candidates[np.argmin(abs(candidates - us_mode_delta))]
            return func.delta_tilde_to_rho(root)

    else:
        # Return the universal profile, which does not depend on mass.
        return us_mode_rho

def most_probable_rho_transformed(m, beta, gamma, sf:float = 1):
    """
        Calculate the most probable rho, correctly transforming delta_tilde -> rho.
        Involves solving the roots of a third-order polynomial.
    """

    # Set up first layer of constants
    delta_c = delta_c_0(sf) * func.D(sf) / func.D(1)
    eta = delta_c_0(sf) - delta_c
    A = func.S(m, gamma) / (2 * func.S(beta * m, gamma) *(func.S(m, gamma) 
                                - func.S(beta * m, gamma)))
    B = delta_c_0(sf) / 2 / (func.S(m, gamma) - func.S(beta * m, gamma))

    # Set up second layer of constants
    Ap = A * pow(delta_c, 2)
    Bp = delta_c * (delta_c * A - B)

    # Set up third layer of constants
    a = 2 * Ap * delta_c
    b = 2 * (Ap * eta - Bp * delta_c)
    c = - (2 * Bp * eta + 2 * delta_c + pow(delta_c, 2))
    d = - eta * (1 + delta_c)

    # Solve the cubic
    try:
        candidates = np.zeros_like(beta)
        for i in range(len(beta)):
            for j in range(len(beta[0])):
                root = poly.polyroots([d, c[i, j], b[i, j], a[i, j]])
                for r in root:
                    if r > 0:
                        candidates[i, j] = pow(r, -delta_c)
        return candidates
    
    except:
        root = poly.polyroots([d, c, b, a])
        candidate = [pow(r, -delta_c) for r in root if r > 0]
        return candidate

"""
    Functions related to the CDF.
"""

def CDF(rho, m, beta, gamma:float = pms.default_gamma, a:float = 1):
    """
        Calculate the analytic CDF as a function of delta_l(rho).
    """

    # Convert rho -> delta_tilde and set constants.
    delta_tilde = func.rho_to_delta_tilde(rho)
    rho_m = pms.Omega_m * pms.rho_c 

    N = (rho_m / m) * 1. / (2 * np.pi * np.sqrt(func.S(beta * m, gamma)) 
        * pow(func.S(m, gamma) - func.S(beta * m, gamma), 3/2))

    A = func.S(m, gamma) / (2 * func.S(beta * m, gamma) *(func.S(m, gamma) - func.S(beta * m, gamma)))
    B = delta_c_0(a) / (2 * (func.S(m, gamma) - func.S(beta * m, gamma)))
    C = (delta_c_0(a) ** 2) / (2 * (func.S(m, gamma) - func.S(beta * m, gamma)))

    # Calculate the CDF in layers.
    cdf_temp = np.sqrt(np.pi / A) * (delta_c_0(a) - B / A) / 2. # 0.5 * np.sqrt(np.pi / A) * (delta_c_0(a) - B / (2 * A))
    cdf_temp *= np.exp(B**2 / A - C) * (1 - erf((B - A * delta_tilde) / np.sqrt(A))) # -1 np.sqrt(A) * delta_tilde - B / np.sqrt(A)
    cdf_temp += np.exp(-A * (delta_tilde**2) + 2 * B * delta_tilde - C) / (2 * A)
    cdf_temp *= N

    return cdf_temp

def conditional_CDF(rho, m, beta, gamma:float = pms.default_gamma, a:float = 1):
    """
        Calculate the normalised analytic CDF as a function of delta_l(rho).
    """
    norm = (CDF(pms.rho_tilde_max, m, beta, gamma, a) 
            - CDF(pms.rho_tilde_min, m, beta, gamma, a))
    return CDF(rho, m, beta, gamma, a) / norm

def n_CDF(pdf, x_vals, x):
    """
        Find the numerical CDF given a numerical PDF and axis.
    """

    i = np.argmin(np.abs(x_vals - x))
    return sum(pdf[:(i+1)])

"""
    Functions for calculating sample statistics from the distribution directly.
"""

def n_modes_variances(pdf : NDArray, rho_vals : NDArray):
    """
        Calculate the mode of the double distribution 
        (i.e. the most probable profile)
        and the standard deviation of the mode, sliced at m.
    """

    try:
        sample_mode = rho_vals[np.argmax(pdf, axis=1)]
        # for r in rho_vals, multiply PDF here times squared diff off sample_mode here
        sample_mode_variance = np.array([[(pdf[j,:,i] * pow(rho_vals 
                                            - sample_mode[j, i], 2)).sum() 
                                            for i in range(len(sample_mode[0]))] 
                                            for j in range(len(sample_mode))])
        sample_mode_variance /= (pms.num_rho - 1)
    
    except:
        sample_mode = rho_vals[np.argmax(pdf)]
        sample_mode_variance = np.array([(pdf[i] * pow(rho_vals 
                                            - sample_mode, 2)).sum() 
                                            for i in range(len(pdf))])
        sample_mode_variance /= (pms.num_rho - 1)

    return sample_mode, np.sqrt(sample_mode_variance)

def n_quantiles(pdf, x_range):
    """
        Calculate the IQR numerically, on a numeric PDF with axis.
    """

    def find_quantile(zscore):
        sm = 0
        # Track the CDF, and iqrs found
        stat = np.zeros_like(pdf[:,0,:])

        # Use the numerical CDF to find the iqrs
        for idx, x in enumerate(x_range):
            sm += pdf[:,idx,:]
            sm_l = np.argwhere(sm > zscore)
            if sm_l.size != 0:
                for i in range(len(sm_l)):
                    f = sm_l[i, 0]
                    s = sm_l[i, 1]
                    if stat[f, s] == 0:
                        stat[f, s] = x
        return stat
    
    return find_quantile(0.5), find_quantile(pms.lqr), find_quantile(pms.uqr)

"""
def solve_combined(args):
    b, m, g, z = args
    cdf_diff = lambda x: abs(conditional_CDF(x, m, b, pms.default_gamma, 1) - z)
    return minimize(cdf_diff, g, bounds=[(pms.rho_tilde_min, pms.rho_tilde_max)],
                        tol=pms.root_finder_precision)

def a_median_and_IQR(sample_mode, sample_stdev, mass, beta,
                 gamma:float = pms.default_gamma, a:float = 1):
    def find_median_and_IQR(q):
        if q == "l": 
            zscore = pms.lqr
            guess = sample_mode - sample_stdev
        elif q == "m":
            zscore = 0.50
            guess = sample_mode
        elif q == "h":
            zscore = pms.uqr
            guess = sample_mode + sample_stdev
        else:
            print("Quantile specified cannot be computed.")
            exit()

        params = [
            (b, m, guess[i, j], zscore)
            for i, b in enumerate(beta)
            for j, m in enumerate(mass)
        ]

        print(f"solving in parallel")
        start = time()
        with Pool(cpu_count()) as pool:
            results = pool.map(solve_combined, params)
        diff = time() - start
        print(f"Loop with {cpu_count()} threads took {diff} seconds.")
        # exit()

        print("extracting solution")
        # Now reconstruct solutions array from flat list 'results'
        solutions = np.zeros_like(guess)
        for idx, res in enumerate(results):
            i = idx // pms.num_mass
            j = idx % pms.num_mass
            # Minimize may return OptimizeResult, get best guess (min location)
            solutions[i, j] = res.x if hasattr(res, 'x') else res

        return solutions

    return find_median_and_IQR("m") #, find_median_and_IQR("l"), find_median_and_IQR("h")

"""