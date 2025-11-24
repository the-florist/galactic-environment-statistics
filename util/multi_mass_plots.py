"""
    Filename: multi-mass-plots.py
    Author: Ericka Florio
    Created: 22 Oct 2025
    Description: -----
"""

import numpy as np
from time import time
from datetime import timedelta
import matplotlib.pyplot as plt

import util.parameters as pms
import src.double_distribution_functions as ddfunc

rho_vals = np.linspace(pms.rho_tilde_min, pms.rho_tilde_max, pms.num_rho)
mass_vals = np.linspace(pms.mass_min, pms.mass_max, pms.num_mass)

slices = np.array([2, 3, 4])

def run():
    mode_diffs0 = np.zeros(shape=(3, pms.num_mass))
    mode_diffs1 = np.zeros(shape=(3, pms.num_mass))
    mode_diffs2 = np.zeros(shape=(3, pms.num_mass))
    b = pms.beta_heuristic

    # Compute un-normalised joint PDF
    for bi, b in enumerate(slices):
        print(r"Starting plot for beta = "+str(b))

        for mi, m in enumerate(mass_vals):
            cond_PDF = np.zeros(pms.num_rho)
            for ri, r in enumerate(rho_vals):
                cond_PDF[ri] = ddfunc.dn(r, m, b, transform_pdf=True)
        
            # Calculate the mode of the PDF numerically and analytically
            numeric_mode, _ = ddfunc.pdf_sample_expectation(cond_PDF, rho_vals)
            full_analytic_mode = ddfunc.most_probable_rho_transformed(m, b)
            us_analytic_mode = ddfunc.most_probable_rho(b)
            us_analytic_mode_with_mass = ddfunc.most_probable_rho(b, inc_mass_scaling=True, m=m)

            mode_diffs0[bi][mi] = ((us_analytic_mode_with_mass - us_analytic_mode)
                                    / us_analytic_mode_with_mass)

            # Take the absolute difference, to be plotted later
            mode_diffs1[bi][mi] = ((us_analytic_mode_with_mass - full_analytic_mode) 
                                    / us_analytic_mode_with_mass)
            mode_diffs2[bi][mi] = ((us_analytic_mode_with_mass - numeric_mode) 
                                / us_analytic_mode_with_mass)
            
            if m == 1e14 or m == 1e15:
                print(f"Mass: {m:.2E}")
                print(f"US mode: {us_analytic_mode:.6E}")
                print(f"US mode with mass: {us_analytic_mode_with_mass:.6E}")
                print(f"Numeric mode corresponding: {rho_vals[np.argmax(cond_PDF)]:.6E}")
                print(f"Full mode: {full_analytic_mode:.6E}")
                print("------")

        # Plot difference between analytic and numeric modes.
        line, = plt.plot(mass_vals, mode_diffs1[bi])
        plt.plot(mass_vals, mode_diffs2[bi], color=line.get_color(), 
                linestyle='--', linewidth=1, label='_nolegend_')

    plt.xlabel(r"$m$")
    plt.ylabel(r"($\hat{\rho}_{us} - \hat{\rho}_{full})/\hat{\rho}_{us}$")
    plt.title("Abs diff between cubic and universal-scaling modes")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/mode-diffs.pdf")
    plt.close()

    for bi, b in enumerate(slices):
        plt.plot(mass_vals, mode_diffs0[bi], label=rf"$beta = {b:.2e}$")
    
    plt.xlabel(r"$m$")
    plt.ylabel(r"($\hat{\rho}_{us,m} - \hat{\rho}_{us})/\hat{\rho}_{us,m}$")
    plt.title("Universal-scaling mode error, excluding mass dependence")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/mode-diffs-excl-mass-dependence.pdf")
    plt.close()