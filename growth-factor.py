import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

Omega_m = float(input("Enter Omega_m (0, 1): "))
Omega_L = float(input("Enter Omega_L (0, 1): "))
a_i = float(input("Enter a scale factor to evaluate at: "))

def integrand(x):
    return pow(x / (x * (1 - Omega_m - Omega_L) + Omega_m + Omega_L * (x ** 3)), 3/2)

out = integrate.quad(lambda x: integrand(x), 0, a_i)
D_i = out[0]
err = out[1]

D_i *= np.sqrt(a_i * (1 - Omega_m - Omega_L) + Omega_m + Omega_L * (a_i ** 3)) / pow(a_i, 3/2)

print("D(a) = "+str(D_i))
print("D(a) error: "+str(err))

plot_factor = bool(input("Plot D(a) from (0, 1)? "))

if(plot_factor):
    def D(a, floor):
        out_full = integrate.quad(lambda x: integrand(x), floor, a)[0] 
        out_full *= np.sqrt(a * (1 - Omega_m - Omega_L) + Omega_m + Omega_L * (a ** 3))
        return out_full

    lowest_a = 0.01
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