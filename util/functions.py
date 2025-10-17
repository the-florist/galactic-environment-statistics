"""
    Filename: functions.py
    Author: Ericka Florio
    Created: 8th September 2025
    Description: Shared functions for calculating the most probable density profile.
"""

# libraries
from math import exp
import string
from numpy.typing import NDArray
import scipy.integrate as integrate
from scipy.optimize import minimize
from scipy.special import erf
import numpy as np
from typing import overload, Literal, Tuple, Union
import os

# parameters
import util.parameters as pms


"""
    Functions used in growth-factor, to calculate D(a)
"""

def make_directory(output_dir):
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

x_of_a = lambda a: pow(2 * pms.w, 1/3) * a
A_integrand = lambda u: pow(u / (pow(u, 3) + 2), 3/2)

def A(x_val):
        out = integrate.quad(A_integrand, x_of_a(pms.a_i), x_val)
        A_tmp = out[0]
        A_tmp *= np.sqrt(pow(x_val, 3) + 2) / pow(x_val, 3/2)
        return A_tmp

def D_integrand(x: float, Om_integrand: float, Ol_integrand: float) -> float:
    return pow(x / (x * (1 - Om_integrand - Ol_integrand) + Om_integrand + Ol_integrand * (x ** 3)), 3/2)

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


def S(m, gamma:float = 0.52, power_law_approx = pms.power_law_approx):
    """
        Variance of the density field 
        calculated both in the power law approximation
        and by the transfer function from Bardeen 1986.
    """
    if power_law_approx == True:
        return pms.s_8 * (m/pms.m_8) ** (-gamma)

    else:
        S_temp = integrate.quad(lambda k: transfer_function_integrand(k), 0, k_of_m(m))[0]
        S_temp *= pms.s_8
        S_temp /= (integrate.quad(lambda k: transfer_function_integrand(k), 0, k_of_m(pms.m_8))[0])
        return S_temp 

def rho(beta, delta_c, gamma:float = 0.52, a = 1, m:float = pms.M_200):
    """
        Find rho(beta) for power law approximation of S(m),
        or rho(beta, m) for transfer function version of S(m).
    """
    if pms.power_law_approx == True:
        return (pms.Omega_m * pms.rho_c) * (a ** -3) * pow(1 - pow(beta, -gamma), -delta_c + 1) / (pms.Omega_m * pms.rho_c)

    else:
        C = pms.s_8 / (integrate.quad(lambda k: transfer_function_integrand(k), 0, k_of_m(pms.m_8)))[0]
        temp = delta_c * pow(1 - S(beta * m)/S(m), -1) * k_of_m(beta * m) * transfer_function_integrand(k_of_m(m)) / 3 / S(m)
        denominator = 1 - temp * C
        return rho_avg(S(beta * m), S(m), delta_c) / denominator
        

def r(beta, delta_c, delta_ta, gamma:float = 0.52, m:float = pms.M_200):
    """
        Find r(beta) for power law approximation of S(m), where r = R/R_ta,
        or r(beta, m) for transfer function version of S(m).
    """
    if pms.power_law_approx == True:
            w = (1 + delta_ta) * pow(1 - pow(1 + delta_ta, -1/delta_c), 1/gamma)
            return pow(w * beta * pow(1 - pow(beta, -gamma), delta_c), 1/3)

    else:
        temp = beta * pow(1 - S(beta * m)/S(m), delta_c) * (1 + delta_ta) / pms.beta_ta # FIXME
        return pow(temp, 1/3)

"""
    Functions used in double-dsitribution.py, to find the double distribution.
"""

def a_coll_integrand(x, c1, c2) -> float:
    return np.sqrt(x / (c1 * x**3 - c2 * x + 1))

def a_coll() -> float:
    """
        Calculate a_coll by minimizing the integral to a_coll and the integral to a_pta
    """
    a_init = 1

    a_pta = pow(pms.w, -1/3) 
    a_pta *= np.sqrt(4 * pms.kappa / 3 / pow(pms.w, 1/3)) 
    a_pta *= np.cos(1/3 * (np.arccos(np.sqrt(27 * pow(pms.kappa/pow(pms.w, 1/3), -3) / 4)) + np.pi))

    C = 2 * integrate.quad(lambda x: a_coll_integrand(x, pms.w, pms.kappa), pms.a_i, a_pta)[0]

    def diff(a):
        return abs(integrate.quad(lambda x: a_coll_integrand(x, pms.w, pms.phi), pms.a_i, a)[0] - C)

    solution = minimize(diff, a_init, bounds=[(0, 1)], tol=pms.root_finder_precision)

    return solution.x[0]

def delta_c_0(a_i : float) -> float:
    a_c = a_coll()
    delta_c = 3 * pms.Omega_m * (pms.kappa - pms.phi) * D(a_c) / 2
    temp = D(1) * delta_c / D(a_i)
    return temp

def CDF(rho, beta:float = 1.3, m:float = pms.M_200, a:float = 1):
    """
        Calculate the CDF as a function of rho (delta_l)
    """
    delta = rho - 1
    delta_c = delta_c_0(a) * D(a) / D(1)
    delta_tilde = delta_c * (1 - pow(1 + delta, -1/delta_c))
    rho_m = pms.Omega_m * pms.rho_c 

    N = (rho_m / m) * 1. / (2 * np.pi * np.sqrt(S(beta * m)) * pow(S(m) - S(beta * m), 3/2))

    A = S(m) / (2 * S(beta * m) *(S(m) - S(beta * m)))
    B = delta_c_0(a) / 2 / (S(m) - S(beta * m))
    C = (delta_c_0(a) ** 2) / (2 * (S(m) - S(beta * m)))

    cdf_temp = np.sqrt(np.pi / A) * (delta_c_0(a) - B / A) / 2. # 0.5 * np.sqrt(np.pi / A) * (delta_c_0(a) - B / (2 * A))
    cdf_temp *= np.exp(B**2 / A - C) * (1 - erf((B - A * delta_tilde) / np.sqrt(A))) # -1 np.sqrt(A) * delta_tilde - B / np.sqrt(A)
    cdf_temp += np.exp(-A * (delta_tilde**2) + 2 * B * delta_tilde - C) / (2 * A)
    cdf_temp *= N

    return cdf_temp

def dn(beta, rho, m:float = pms.M_200, a:float = 1):
    """
        Calculate the double distribution of number density w/r/t mass and local overdensity
    """

    delta = rho - 1
    delta_c = delta_c_0(a) * D(a) / D(1)
    delta_tilde = delta_c * (1 - pow(1 + delta, -1/delta_c))
    rho_m = pms.Omega_m * pms.rho_c 

    mass_removal = (delta_c_0(a) - delta_tilde) * np.exp(-(delta_c_0(a) - delta_tilde)**2 / (2 *(S(m) - S(beta*m)))) 
    mass_removal /= pow(S(m) - S(beta*m), 3/2)

    random_walk = (rho_m / m) * (np.exp(-(delta_tilde ** 2) / (2 * S(beta * m))) / (2 * np.pi * np.sqrt(S(beta*m))))

    jacobian = pow(rho, (-1 - 1/delta_c))

    dn = random_walk * mass_removal * jacobian

    if pms.enforce_positive_pdf == True:
        if dn < 0:
            return 0
        else:
            return dn
    else:
        return dn

# def pdf_analytic_expectation(m : float, beta : float, a : float = 1):
#     """
#         Calculate the expectation value of the double distribution sliced at m.
#     """

#     A = S(beta * m)
#     B = delta_c_0(a)
#     C = S(m)

#     temp = A * (A - C) * (-B**2 + C) * np.exp(-B**2/2/C)
#     temp /= (C**2 * np.sqrt(-C/(2 * A**2 * np.pi - 2 * A * C * np.pi)))

#     norm = B * (-A+C) * np.exp(-B**2/2/C)
#     norm /= (C * np.sqrt(-C/(2 * A**2 * np.pi - 2 * A * C * np.pi)))

#     return norm, temp/norm

def pdf_sample_expectation(pdf : list, rho_vals : NDArray):
    """
        Calculate the mode of the double distribution (i.e. the most probable profile)
        and the standard deviation of the mode, sliced at m.
    """

    sample_mode = rho_vals[pdf.index(max(pdf))]

    sample_mode_variance = 0
    for i in range(len(rho_vals)):
        sample_mode_variance += pdf[i] * pow(rho_vals[i] - sample_mode, 2) # / sample_norm

    return sample_mode, np.sqrt(sample_mode_variance)

def most_probable_rho(beta:float, gamma:float = 0.52, a:float = 1):
    delta_c = delta_c_0(a) * D(a) / D(1)
    return pow(1 - pow(beta, -gamma), -delta_c + 1)

"""
def convergence_test(my_analytic_function, my_analytic_diagnostic, my_sample_diagnostic, param_1, param_2, 
                        my_min : float, my_max : float, resolutions : list):

        # Ensure output directory exists
        make_directory("output")
        output_path = os.path.join(output_dir, "convergence-test.txt")
        fixed_point, _ = my_analytic_diagnostic(param_1, param_2)
        with open(output_path, "w") as f:
            f.write("res\tfixed_point\tguess\tnorm_diff\n")
            for res in resolutions:
                xs = np.linspace(my_min, my_max, res)
                data = [my_analytic_function(x, param_1, param_2) for x in xs]
                guess, _ = my_sample_diagnostic(data, xs)
                # If sample diagnostic returns tuple, take first element
                if isinstance(guess, tuple):
                    guess = guess[0]
                norm = max(abs(fixed_point), abs(guess))
                norm_diff = abs(guess - fixed_point) / norm if norm != 0 else 0
                f.write(f"{res}\t{fixed_point:.8e}\t{guess:.8e}\t{norm_diff:.3e}\n")
"""

def IQR(beta, guess_l, guess_h, m:float = pms.M_200/pms.m_8, a:float = 1):
    """
        Calculate the IQR at a slice in beta using a root finding method.
    """
    
    cdf_slice = lambda rho : CDF(rho, beta, m, a)
    diff_l = lambda rho : cdf_slice(rho) - 0.25
    diff_h = lambda rho : cdf_slice(rho) - 0.75

    iqr_l = minimize(diff_l, guess_l, bounds=[(1, 1e6)], tol=pms.root_finder_precision)
    iqr_h = minimize(diff_h, guess_h, bounds=[(1, 1e6)], tol=pms.root_finder_precision)

    return iqr_l.x[0], iqr_h.x[0]