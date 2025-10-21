"""
    Filename: parameters.py
    Author: Ericka Florio
    Created: 4th September 2025
    Description: Parameters for the growth-factor.py program.
    Note: Cosmological parameters in each case match those used by 
        Pavlidou & Fields, PRD 71 043510 (2005), page 7
"""

# Scale factor range
a_f = 1
a_i = 0.01                  # Avoids divide by zero in integration
num_steps = 200

# Flags for growth-factor.py
print_D_a = True
compare_case_1 = True      # Einstein-de-Sitter
compare_case_2 = True      # Flat Universe

# Parameters and flags for density-profile.py
power_law_approx = True
default_gamma = 0.52
root_finder_precision = 1e-5 # Precision used for root finding

# Parameters for double-distribution.py
# Both independent vars. are unitless
beta_min, beta_max, num_beta = 1.5, 5, 25
rho_tilde_min, rho_tilde_max, num_rho = 0, 3, 100 # 60

plot_dimension = 1
enforce_positive_pdf = True
# run_convergence_tests = False

# Concordance cosmology
concordance_model = True

#####################
#####################
#####################

if concordance_model:
    Omega_m = 0.27
    Omega_L = 1 - Omega_m
    w = Omega_L / Omega_m
    phi = (Omega_m + Omega_L - 1)/Omega_m
    kappa = 1.50 * 3 * pow(w, 1/3) / pow(2, 2/3)
    s_8 = 0.84
    m_8 = 2e14    
    delta_c = 1.6757          # Solar masses

# Einstein de-Sitter
else:
    Omega_m = 1
    w = 0
    phi = 0
    kappa = 1.50 * 3 * pow(w, 1/3) / pow(2, 2/3)
    s_8 = 0.45
    m_8 = 8e14   
    delta_c = 1.6865           # Solar masses

###############

# Fixed cosmological parameters (do not change!)
h = 71 / 100                # unitless H0
rho_c = 2.78e11 * (h ** 2)  # units of Solar Mass / Mpc^3
n = 1                       # primordial spectral index
M_200 = 6.3e14              # virial mass scale