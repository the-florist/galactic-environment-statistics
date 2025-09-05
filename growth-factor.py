import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Import parameters
import parameters as mp

# Define functions
def integrand(x, Om = mp.Omega_m, Ol = mp.Omega_L):
    return pow(x / (x * (1 - Om - Ol) + Om + Ol * (x ** 3)), 3/2)

def D(a, floor, return_full = False, Om = mp.Omega_m, Ol = mp.Omega_L):
    out_full = integrate.quad(lambda x: integrand(x), floor, a)

    D_temp = out_full[0]
    D_temp *= np.sqrt(a * (1 - Om - Ol) + Om + Ol * (a ** 3)) / pow(a, 3/2)
    err = out_full[1]
    
    if(return_full):
        return [D_temp, err]
    else:
        return D_temp

# Compute D(a) for a single value
out = D(mp.a_f, mp.a_i, return_full = True)
D_i = out[0]
err = out[1]
print("D(a) = "+str(D_i))
print("D(a) error: "+str(err))

a_vals = np.linspace(mp.a_i, 1, mp.num_steps)
D_vals = [D(a, mp.a_i) for a in a_vals]

# Plot D(a)
if(mp.print_D_a == True):
    plt.plot(a_vals, D_vals, label="D(a)")
    plt.xlabel("a")
    plt.ylabel("D(a)")
    plt.title("Growth Factor D(a) vs Scale Factor a")
    plt.legend()
    plt.grid(True)
    plt.savefig("D-plot.pdf")

if(mp.compare_case_1 == True):
    matter_model = [D(a, mp.a_i, Om = 1, Ol = 0) for a in a_vals]
    slope_est = (matter_model[-1] - matter_model[0])/(mp.a_f - mp.a_i)
    
    plt.plot(a_vals, matter_model, label="D(a), Omega_m = 1")
    plt.plot(a_vals, slope_est * a_vals, label="Linear in a", linestyle='--')
    plt.xlabel("a")
    plt.ylabel("D(a)")
    plt.title("Growth Factor D(a) vs Scale Factor a")
    plt.legend()
    plt.grid(True)
    plt.savefig("D-plot-compare-case-1.pdf")

print("Program finished.")