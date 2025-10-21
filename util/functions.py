"""
    Filename: functions.py
    Author: Ericka Florio
    Created: 8th September 2025
    Description: Shared functions for calculating the most probable density profile.
"""

# libraries
import os
import numpy as np
import scipy.integrate as integrate
from typing import overload, Literal, Tuple, Union

# parameters
import util.parameters as pms

"""
    General purpose functions
"""

def make_directory(output_dir):
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

def clear_file(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)

"""
    Functions used to calculate the growth factor D(a)
"""

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


"""
    Functions used to calculate the matter variance S(m)
"""

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


def S(m, power_law_approx = pms.power_law_approx, gamma:float = pms.default_gamma):
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
        
"""
    Functions yet to be implimented
"""

"""
def convergence_test(my_analytic_function, my_analytic_diagnostic, my_sample_diagnostic, param_1, param_2, 
                        my_min : float, my_max : float, resolutions : list):

        # Ensure output directory exists
        make_directory("output")
        output_path = "output/convergence-test.txt"
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

"""
def IQR(beta, guess_l, guess_h, m:float = pms.M_200/pms.m_8, a:float = 1):
    
    cdf_slice = lambda rho : CDF(rho, beta, m, a)
    diff_l = lambda rho : cdf_slice(rho) - 0.25
    diff_h = lambda rho : cdf_slice(rho) - 0.75

    iqr_l = minimize(diff_l, guess_l, bounds=[(1, 1e6)], tol=pms.root_finder_precision)
    iqr_h = minimize(diff_h, guess_h, bounds=[(1, 1e6)], tol=pms.root_finder_precision)

    return iqr_l.x[0], iqr_h.x[0]
"""