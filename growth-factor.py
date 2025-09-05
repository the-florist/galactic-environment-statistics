import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Set parameters
Omega_m = float(input("Enter Omega_m (0, 1): "))
Omega_L = float(input("Enter Omega_L (0, 1): "))
a_i = float(input("Enter a scale factor to evaluate at: "))
lowest_a = 0.01

# Define functions
def integrand(x):
    return pow(x / (x * (1 - Omega_m - Omega_L) + Omega_m + Omega_L * (x ** 3)), 3/2)

def D(a, floor, return_full = False):
    out_full = integrate.quad(lambda x: integrand(x), floor, a)

    D_temp = out_full[0]
    D_temp *= np.sqrt(a * (1 - Omega_m - Omega_L) + Omega_m + Omega_L * (a ** 3))
    err = out_full[1]
    
    if(return_full):
        return [D_temp, err]
    else:
        return D_temp

# Compute D(a) for a single value
out = D(a_i, lowest_a, return_full = True)
D_i = out[0]
err = out[1]

print("D(a) = "+str(D_i))
print("D(a) error: "+str(err))

# Plot D(a)
plot_factor = bool(input("Plot D(a) from (0, 1)? "))
if(plot_factor == True):
    a_vals = np.linspace(lowest_a, 1, 200)
    D_vals = [D(a, lowest_a) for a in a_vals]

    plt.plot(a_vals, D_vals, label="D(a)")
    plt.xlabel("a")
    plt.ylabel("D(a)")
    plt.title("Growth Factor D(a) vs Scale Factor a")
    plt.legend()
    plt.grid(True)
    plt.savefig("D-plot.pdf")


print("Program finished.")