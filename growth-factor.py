"""
    Filename: growth-factor.py
    Author: Ericka Florio
    Created: 4th September 2025
    Description: Calculation of the linear transfer function D(a):
        - at a particular a given in parameters,
        - as a function of a, for Omega_m = 1 (Einstein-de-Sitter),
        - and as a function of a for Omega_m + Omega_L = 1 (flat Universe)
"""

# libraries
import matplotlib.pyplot as plt
import numpy as np
from typing import List, cast

# custom files
import util.parameters as pms
import util.functions as func

# Compute D(a) for a single value
D_i, err = func.D(pms.a_f, return_full = True)
print("D(a) = "+str(D_i))
print("D(a) error: "+str(err))

a_vals = np.linspace(pms.a_i, pms.a_f, pms.num_steps)
D_vals: List[float] = [cast(float, func.D(a, return_full=False)) for a in a_vals]

# Plot D(a)
if(pms.print_D_a == True):
    plt.plot(a_vals, D_vals, label="D(a)")
    plt.xlabel("a")
    plt.ylabel("D(a)")
    plt.title("Growth Factor D(a) vs Scale Factor a")
    plt.legend()
    plt.grid(True)

    func.make_directory("plots")
    plt.savefig("plots/D-plot.pdf")
    plt.close()

# Check the matter-only case works
if(pms.compare_case_1 == True):
    matter_model: List[float] = [cast(float, func.D(a, return_full=False, Om = 1, Ol = 0)) for a in a_vals]
    slope_est = (matter_model[-1] - matter_model[0])/(pms.a_f - pms.a_i)
    
    plt.plot(a_vals, matter_model, label="D(a), Omega_m = 1")
    plt.plot(a_vals, slope_est * a_vals, label="Linear in a", linestyle='--')
    plt.xlabel("a")
    plt.ylabel("D(a)")
    plt.title("Growth Factor D(a) vs Scale Factor a")
    plt.legend()
    plt.grid(True)

    func.make_directory("plots")
    plt.savefig("plots/D-case-1.pdf")
    plt.close()

# Check the Matter + DM = 1 case works
if(pms.compare_case_2 == True):
    Omega_m = 0.7
    Omega_L = 0.3
    if(Omega_L + Omega_m != 1):
        print(Omega_m, Omega_L, Omega_m + Omega_L)
        raise Exception("For test 2, Omega_m and Omega_L must sum to 1")

    even_evaluation: List[float] = [cast(float, func.D(a, return_full=False, Om = Omega_m, Ol = Omega_L)) for a in a_vals]
    even_model = [func.A(func.x_of_a(a)) for a in a_vals]

    rescale = pow(Omega_L, -3/2) * pow(2 * pms.w, 2/3) * np.sqrt(Omega_L) 
    even_model_rescaled = [(func.A(func.x_of_a(a)) * rescale) for a in a_vals]

    plt.plot(a_vals, even_evaluation, label="D(a)")
    plt.plot(a_vals, even_model, label="A(x)", linestyle='--')
    plt.plot(a_vals, even_model_rescaled, label="A(x) rescaled", linestyle='--')

    plt.xlabel("a")
    plt.ylabel("Growth factor")
    plt.title("Example 2: Omega_m + Omega_L = 1")
    plt.legend()
    plt.grid(True)

    func.make_directory("plots")
    plt.savefig("plots/D-case-2.pdf")
    plt.close()

print("Program finished.")