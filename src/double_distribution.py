"""
    Filename: double-distribution.py
    Author: Ericka Florio
    Created: 11 Sept 2025
    Description: Plots the joint double distribution for the number density of objects 
            of mass m with local overdensity delta_l, 
            as derived in Pavlidou and Fields 2005.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import util.parameters as pms
import util.functions as func
import src.double_distribution_functions as ddfunc

print("Plotting the double distribution.")
if pms.plot_dimension != 1 and pms.plot_dimension != 2:
    raise ValueError("double-distribution.py : plot_dimension impossible" 
                     "or is not implemented.")

beta_vals = np.logspace(np.log10(pms.beta_min), np.log10(pms.beta_max), pms.num_beta)
lin_rho_vals = np.linspace(pms.rho_tilde_min, pms.rho_tilde_max, pms.num_rho)
rho_vals = np.array([pow(10, r) for r in lin_rho_vals])
dr = (pms.rho_tilde_max - pms.rho_tilde_min) / (pms.num_rho)

print("-----\nDomain: ")
print(rf"Mass range: ({pms.beta_min:.1e}, {pms.beta_max:.1e})")
print("Density contrast range: ("+str(pms.rho_tilde_min)+", "
      +str(pms.rho_tilde_max)+")")

def run():

    if pms.plot_dimension == 2:
        # Create meshgrid for m and delta_l
        BTS, RHOS = np.meshgrid(beta_vals, rho_vals, indexing='ij')

        print("Starting loop over mass/contrast domain...")

        # Compute joint pdf using dn
        Z = np.zeros_like(BTS)
        for i in range(pms.num_beta):
            for j in range(pms.num_rho):
                try:
                    Z[i, j] = ddfunc.dn(BTS[i, j], RHOS[i, j])
                except Exception:
                    Z[i, j] = 0

        for i in range(pms.num_beta):
            norm = (sum(Z[i]) - Z[i, 0]) * dr
            for j in range(pms.num_rho):
                Z[i, j] /= norm

        # Marginals
        marginal_m = np.trapezoid(Z, rho_vals, axis=1)
        marginal_dl = np.trapezoid(Z, beta_vals, axis=0)

        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)

        ax_joint = fig.add_subplot(gs[1:,:3])
        ax_marg_m = fig.add_subplot(gs[0,:3], sharex=ax_joint)
        ax_marg_dl = fig.add_subplot(gs[1:,3], sharey=ax_joint)

        # Joint pdf heatmap
        c = ax_joint.pcolormesh(beta_vals, rho_vals, Z.T, shading='auto', 
                                                          cmap='viridis')
        ax_joint.set_xlabel(r'$\beta$')
        ax_joint.set_ylabel(r'$\tilde{\rho}$')
        ax_joint.set_xscale('log')
        ax_joint.set_yscale('log')

        # Marginal for m
        ax_marg_m.plot(beta_vals, marginal_m, color='tab:blue')
        ax_marg_m.set_xscale('log')
        ax_marg_m.set_ylabel(r'Marginal ($\beta$)')
        ax_marg_m.tick_params(axis='y', labelbottom=False)
        ax_marg_m.grid(True, which='both', ls='--', alpha=0.3)

        # Marginal for delta_l
        ax_marg_dl.plot(marginal_dl, rho_vals, color='tab:orange')
        ax_marg_dl.set_xlabel(r'Marginal ($\tilde{\rho}$)')
        ax_marg_dl.tick_params(axis='y', labelleft=False)
        ax_marg_dl.grid(True, which='both', ls='--', alpha=0.3)
        ax_marg_dl.xaxis.get_offset_text().set_x(1.2)  # Move right
        ax_marg_dl.xaxis.get_offset_text().set_y(-0.15)  # Move down a bit

        # Remove tick labels on shared axes
        plt.setp(ax_marg_m.get_xticklabels(), visible=False)
        plt.setp(ax_marg_dl.get_yticklabels(), visible=False)

        ax_joint.set_title(r'Joint PDF of $\beta$ and $\tilde{\rho}$ with marginals')

        func.make_directory("plots")
        plt.savefig("plots/joint-pdf.pdf")
        plt.close()

    elif pms.plot_dimension == 1:
        """
            Calculate the PDF at slices of beta, and plot alongside the mode 
            and stdev of the mode.
        """

        func.make_directory("output")
        fname = "output/mpp-info.txt"
        func.clear_file(fname)

        beta_slices = [i * np.floor(pms.num_beta/5) for i in range(5)]
        num_beta = 0
        mode_diffs = []

        for idx, beta in enumerate(beta_vals):
            PDF = [ddfunc.dn(beta, rho) for rho in rho_vals]
            numeric_mode, _ = ddfunc.pdf_sample_expectation(PDF, rho_vals)
            analytic_mode = ddfunc.most_probable_rho(beta)
            # analytic_mode_transformed = func.most_probable_rho_transformed(beta)

            # print(numeric_mode, analytic_mode)
            # print(analytic_mode_transformed)
            # exit()

            mode_diffs.append(abs(numeric_mode - analytic_mode) / numeric_mode)

            num_beta += 1
            with open(fname, "a") as file:
                diff = abs(numeric_mode - analytic_mode) / numeric_mode
                if (num_beta == 1):
                    file.write("beta\tmpr-analytic\tmpr-numeric"
                               "\trelative-difference\n")
                file.write(f"{beta:.4E}\t{analytic_mode}\t{numeric_mode}"
                           f"\t{diff}\n")

            if idx in beta_slices:
                CDF = [sum(PDF[:i]) * dr for i in range(len(PDF))]
                norm = (CDF[-1] - CDF[0])
                for i in range(len(PDF)):
                    PDF[i] /= norm
                sample_norm = sum(PDF) * dr

                print(f"PDF sample norm accuracy: {abs(1-sample_norm)}")
                print(f"Most probable rho, DD: {numeric_mode}")
                print(f"Most probable rho, analytic model: {analytic_mode}")
                print("-----------")

                plt.plot(rho_vals, PDF, label=rf"$\beta$ = {beta:.2}")
                plt.plot(numeric_mode, max(PDF), 'ro', label='_nolegend_')
        
        plt.xlabel(r"$\tilde{\rho}$")
        plt.ylabel(r"$P_n$")
        plt.xscale("log")
        plt.title(r"CDF slices along $\beta$")
        plt.grid(True)
        plt.legend()
        plt.savefig("plots/joint-pdf-slice.pdf")
        plt.close()

        plt.plot(beta_vals, mode_diffs)
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"$|\hat{M} - M|/\hat{M}$")
        plt.title("Absolute difference between sample and predicted modes")
        plt.grid(True)
        plt.savefig("plots/mode-diffs.pdf")
        plt.close()