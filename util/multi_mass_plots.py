"""
    Filename: multi-mass-plots.py
    Author: Ericka Florio
    Created: 22 Oct 2025
    Description: -----
"""

import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import util.parameters as pms
import util.functions as func
import src.double_distribution_functions as ddfunc

beta_vals = np.logspace(np.log10(pms.beta_min), np.log10(pms.beta_max), pms.num_beta)
lin_rho_vals = np.linspace(pms.rho_tilde_min, pms.rho_tilde_max, pms.num_rho)
rho_vals = np.array([pow(10, r) for r in lin_rho_vals])
m_vals = np.array([(1 - pms.mass_percent_diff) * pms.M_200, pms.M_200, 
                   (1 + pms.mass_percent_diff) * pms.M_200])

def run():
    # Create meshgrid for beta and rhos
    BTS, RHOS = np.meshgrid(beta_vals, rho_vals, indexing='ij')

    print("Starting loop over rho/beta domain...")

    for m in m_vals:
        # Compute un-normalised joint PDF
        PDF = np.zeros_like(BTS)

        # Set up the timer
        start = time()
        intv = 15
        last = start

        for i in range(pms.num_beta):
            for j in range(pms.num_rho):
                try:
                    PDF[i, j] = ddfunc.dn(BTS[i, j], RHOS[i, j])
                except Exception:
                    PDF[i, j] = 0

            now = time()
            if now - last > intv:
                frac = ((i * pms.num_rho) + j + 1) / (pms.num_beta * pms.num_rho)
                elapsed = now - start
                print(f"{elapsed:.2g} sec. passed, {frac * 100}% finished...")
                last = now

        # Normalise the joint PDF
        if pms.normalise_pdf:
            norm = np.sum(PDF)
            PDF /= norm
            print(f"PDF norm precision: {abs(np.sum(PDF) - 1.):.15}")

        print(f"Starting {pms.plot_dimension}D plot generation at mass {m}...")

        mode_diffs = []
        for i in range(pms.num_beta):
            # Calculate the mode of the PDF numerically and analytically
            numeric_mode, _ = ddfunc.pdf_sample_expectation(PDF[i], rho_vals)
            analytic_mode = ddfunc.most_probable_rho(beta_vals[i])

            # Take the absolute difference, to be plotted later
            diff = abs(numeric_mode - analytic_mode) / numeric_mode
            mode_diffs.append(diff)

        # Plot difference between analytic and numeric modes.
        plt.plot(beta_vals, mode_diffs)
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"$|\hat{M} - M|/\hat{M}$")
        plt.title("Absolute difference between sample and predicted modes")
        plt.grid(True)
        plt.savefig("plots/mode-diffs.pdf")
        plt.close()