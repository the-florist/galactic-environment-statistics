"""
    Author: Ericka Florio
    Created: 4th September 2025
    Description: Parameters for the growth-factor.py program.
"""

# Cosmological model parameters
Omega_m = 0.3
Omega_L = 0.7
w = Omega_L / Omega_m

# Scale factor range
a_f = 1
a_i = 0.01                 # Avoids divide by zero in integration
num_steps = 200

print_D_a = True

# For growth-factor.py, compare to two known solutions?
compare_case_1 = True      # Einstein-de-Sitter
compare_case_2 = True      # Flat Universe

# Fixed cosmological parameters
# Measured values from the Planck 2018 data release
h = 67.3 / 100              # unitless H0
rho_c = 2.78e11 * (h ** 2)  # units of Solar Mass / Mpc^3
s_8 = 0.811                 # Sigma 8

# For density-profile.py, use universal scaling or not
power_law_approx = False
m_8 = 1 # FIXME
n = 1 # spectral index
beta_ta = 1 # FIXME