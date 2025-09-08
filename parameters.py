"""
    Author: Ericka Florio
    Created: 4th September 2025
    Description: Parameters for the growth-factor.py program.
"""

Omega_m = 0.2
Omega_L = 0.8
w = Omega_L / Omega_m

a_f = 1
a_i = 0.01                 # Avoids divide by zero in integration
num_steps = 200

print_D_a = True

compare_case_1 = True      # Einstein-de-Sitter
compare_case_2 = True      # Flat Universe