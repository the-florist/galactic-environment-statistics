import numpy as np
import matplotlib as mpl
import scipy.integrate as integrate
import sys

Omega_m = float(input("Enter Omega_m (0, 1): "))
Omega_L = float(input("Enter Omega_L (0, 1): "))
a_i = float(input("Enter a scale factor to evaluate at: "))

def integrand(x, O_m, O_l):
    return pow(x / (x * (1 - O_m - O_l) + O_m + O_l * (x ** 3)), 3/2)

out = integrate.quad(lambda x: integrand(x, Omega_m, Omega_L), 0, a_i)
D = out[0]
err = out[1]

D *= np.sqrt(a_i * (1 - Omega_m - Omega_L) + Omega_m + Omega_L * (a_i ** 3)) / pow(a_i, 3/2)

print("D(a) = "+str(D))
print("D(a) error: "+str(err))

print("Program finished.")