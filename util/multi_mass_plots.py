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

beta_vals = np.logspace(np.log10(pms.beta_min), np.log10(pms.beta_max), pms.num_beta)
lin_rho_vals = np.linspace(pms.rho_tilde_min, pms.rho_tilde_max, pms.num_rho)
rho_vals = np.array([pow(10, r) for r in lin_rho_vals])
m_vals = np.array([(1 - pms.mass_percent_diff) * pms.M_200, pms.M_200, 
                   (1 + pms.mass_percent_diff) * pms.M_200])

def run():
    # Create meshgrid for beta and rhos
    BTS, RHOS = np.meshgrid(beta_vals, rho_vals, indexing='ij')
    mode_diffs = np.zeros(shape=(3, pms.num_beta))

    for mi, m in enumerate(m_vals):
        # Compute un-normalised joint PDF
        PDF = np.zeros_like(BTS)

        # Set up the timer
        start = time()
        intv = 15
        last = start

        print("Starting loop over rho/beta domain...")

        for bi in range(pms.num_beta):
            for ri in range(pms.num_rho):
                try:
                    PDF[bi, ri] = ddfunc.dn(BTS[bi, ri], RHOS[bi, ri], m)
                except Exception:
                    PDF[bi, ri] = 0

            now = time()
            if now - last > intv:
                frac = ((bi * pms.num_rho) + ri + 1) / (pms.num_beta * pms.num_rho)
                elapsed = timedelta(seconds=(now - start))
                print(f"{str(elapsed)} sec. passed,"
                      f" {(frac * 100):.2g}% finished...")
                last = now

        # Normalise the joint PDF
        if pms.normalise_pdf:
            norm = np.sum(PDF)
            PDF /= norm
            print(f"PDF norm precision: {abs(np.sum(PDF) - 1.):.15}")

        print(f"Starting mode difference calculation at mass {m:.2E}...")

        for bi in range(pms.num_beta):
            # Calculate the mode of the PDF numerically and analytically
            numeric_mode, _ = ddfunc.pdf_sample_expectation(PDF[bi], rho_vals)
            analytic_mode = ddfunc.most_probable_rho(beta_vals[bi])

            # Take the absolute difference, to be plotted later
            diff = abs(numeric_mode - analytic_mode) / analytic_mode
            mode_diffs[mi][bi] = diff

    # Plot difference between analytic and numeric modes.
    for mi, m in enumerate(m_vals):
        plt.plot(beta_vals, mode_diffs[mi], label=rf"$m = {m:.2e}$")
        # print(mode_diffs[mi])
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$|\hat{M} - M|/\hat{M}$")
    plt.title("Absolute difference between sample and predicted modes")
    plt.grid(True)
    plt.legend()
    plt.savefig("plots/mode-diffs.pdf")
    plt.close()