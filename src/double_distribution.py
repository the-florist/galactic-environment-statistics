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
import util.double_distribution_functions as ddfunc
from util.Newton_method import NewtonsMethod
from src.double_distribution_calculations import DoubleDistributionCalculations

if pms.plot_dimension != 1 and pms.plot_dimension != 2:
    raise ValueError("double-distribution.py : plot_dimension impossible" 
                     "or is not implemented.")

beta_vals = np.linspace(pms.beta_min, pms.beta_max, pms.num_beta)
rho_vals = np.linspace(pms.rho_tilde_min, pms.rho_tilde_max, pms.num_rho)
mass_vals = np.linspace(pms.mass_min, pms.mass_max, pms.num_mass) # np.array([1.3, 5, 17]) * 1e14  # np.array([pms.M_200])
gamma_slices = np.array([0.4]) # np.array([0.4, 0.5, 0.6]) # np.linspace(pms.gamma_min, pms.gamma_max, pms.num_gamma)


def run():
    # Create meshgrid with axes ordered as (beta, rho, mass)
    BTS, RHOS, MS = np.meshgrid(beta_vals, rho_vals, mass_vals, indexing='ij')

    print(f"Starting {pms.plot_dimension}D plot generation...")

    if pms.plot_dimension == 2:
        """
            Calculate the normalised joint PDF and plot it.
        """

        ddc = DoubleDistributionCalculations()
        PDF, norm = ddc.calc_PDF(True, pms.default_gamma)

        b = np.abs(beta_vals - pms.beta_heuristic).argmin()

        # Marginals
        marginal_m = PDF[b].sum(axis=0)
        marginal_rho = PDF[b].sum(axis=1)

        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(4, 4, hspace=0.05, wspace=0.05)

        ax_joint = fig.add_subplot(gs[1:,:3])
        ax_marg_m = fig.add_subplot(gs[0,:3], sharex=ax_joint)
        ax_marg_dl = fig.add_subplot(gs[1:,3], sharey=ax_joint)

        # Joint pdf heatmap
        c = ax_joint.pcolormesh(rho_vals, mass_vals / 1e14, PDF[b].T, shading='auto',
                                                          cmap='viridis')
        ax_joint.set_xlabel(r'$\rho/\rho_m$')
        ax_joint.set_ylabel(r'$m\ [10^{14} M_{\odot}]$')

        ax_joint.ticklabel_format(style='plain')

        # Make mass-axis (y) scientific offset (e.g., 1e15) vertical on the left
        y_off = ax_joint.yaxis.get_offset_text()
        y_off.set_rotation(90)
        y_off.set_verticalalignment('bottom')
        y_off.set_horizontalalignment('right')
        y_off.set_x(-0.1)
        y_off.set_clip_on(False)

        # Marginal for m
        ax_marg_m.plot(rho_vals, marginal_rho, color='tab:blue')
        ax_marg_m.set_ylabel(r'Marginal ($\bar{\rho}$)')
        ax_marg_m.grid(True, which='both', ls='--', alpha=0.3)
        ax_marg_m.set_title(r'Joint PDF of $m$ and $\tilde{\rho}$ at $\beta = $' + str(b), pad=12)

        # Marginal for delta_l
        ax_marg_dl.plot(marginal_m, mass_vals / 1e14, color='tab:orange')
        ax_marg_dl.set_xlabel(r'Marginal ($m$)')
        ax_marg_dl.grid(True, which='both', ls='--', alpha=0.3)

        # Remove tick labels on shared axes
        plt.setp(ax_marg_m.get_xticklabels(), visible=False)
        plt.setp(ax_marg_dl.get_yticklabels(), visible=False)


        func.make_directory("plots")
        plt.savefig("plots/joint-pdf.pdf")
        plt.close()

    elif pms.plot_dimension == 1:
        """
            Calculate the conditional PDF at slices of beta, and plot alongside the mode 
            and stdev of the mode.
        """
        ddc = DoubleDistributionCalculations()

        if pms.slice_in_rho:
            # Numerically construct the conditional PDF
            cond_PDF, norm = ddc.calc_PDF(True, pms.default_gamma)

            # Find the closest beta to our heuristic value
            b = np.abs(beta_vals - pms.beta_heuristic).argmin()

            # Find the analytic most probable mode
            a_mode_transformed = ddfunc.most_probable_rho_transformed(MS[:,0,:], 
                                        BTS[:,0,:], pms.default_gamma)

            # Find the numerical modes, and variances rooted at those modes
            n_modes, n_stdevs = ddfunc.n_modes_variances(cond_PDF, rho_vals)
            
            # Find the median and IQRs analytically
            a_stats = []
            guesses = np.array([n_modes, n_modes - n_stdevs, n_modes + n_stdevs])
            for i, s in np.ndenumerate(np.array([0.5, pms.lqr, pms.uqr])):
                nm = NewtonsMethod(MS[:,0,:], BTS[:,0,:], guesses[i], 
                                pms.default_gamma, s)
                nm.run()
                a_stats.append(nm.return_solution())
            
            # Find the numerical median and IQR
            n_median, n_IQRl, n_IQRu = ddfunc.n_quantiles(cond_PDF, rho_vals)

            if pms.plot_untransformed_PDF:
                # Evaluate the untransformed PDF on the grid
                cond_PDF_nt, norm_nt = ddc.calc_PDF(False, pms.default_gamma)

                # Find the analytic and numeric most probable mode, untransformed
                a_mode_no_transform = ddfunc.most_probable_rho(MS[:,0,:], 
                                             BTS[:,0,:], inc_mass_scaling=True)
                n_mode_nt, _ = ddfunc.n_modes_variances(cond_PDF_nt, rho_vals)
            
            for mi, m in enumerate(mass_vals):
                # Uncomment this to bug check the median
                # if pms.verbose:
                #     n_cdf = ddfunc.conditional_CDF(n_median[b,mi], m, beta_vals[b])
                #     a_cdf = ddfunc.conditional_CDF(a_stats[0][b,mi], m, beta_vals[b])
                #     print("Median estimates: ", n_median[b,mi], a_stats[0][b,mi])
                #     print("Conditional CDF of each: ", n_cdf, a_cdf)
                #     print("Target fn for each: ")
                #     print(nm.target_fn(n_median[b,mi])[b,mi])
                #     print(nm.target_fn(a_stats[0][b,mi])[b,mi])

                # Plot the PDF
                line, = plt.plot(rho_vals, cond_PDF[b,:,mi], 
                                 label=rf"$m = {MS[b,1,mi]:.2e}$")
                plot_color = line.get_color()

                if pms.plot_statistics:
                    # Plot analytic mode
                    plt.plot(a_mode_transformed[b,mi], 
                             ddfunc.dn(a_mode_transformed[b,mi], m, beta_vals[b], 
                             transform_pdf=True) / norm[b,:,mi], 'o', color='red', 
                             label='__nolabel__')

                    # Plot numeric mode
                    plt.plot(n_modes[b,mi], ddfunc.dn(n_modes[b,mi], m, beta_vals[b], 
                                                transform_pdf=True) / norm[b,:,mi],
                                                "*", color="blue", label='__nolabel__')

                    # Plot analytic median
                    plt.plot(a_stats[0][b,mi], 
                        ddfunc.dn(a_stats[0][b,mi], m, beta_vals[b], 
                                    transform_pdf=True) / norm[b,:,mi], 
                        'o', color='red', label='__nolabel__')
                    
                    # Plot numeric median
                    plt.plot(n_median[b,mi], 
                            ddfunc.dn(n_median[b,mi], m, beta_vals[b], transform_pdf=True) / norm[b,:,mi],
                            "*", color="blue", label='__nolabel__')     

                    # Create logical masks where the PDF lies inside the IQRs
                    a_mask = np.logical_and(rho_vals >= a_stats[1][b,mi], rho_vals 
                                            <= a_stats[2][b,mi]).tolist()
                    n_mask = np.logical_and(rho_vals >= n_IQRl[b,mi], rho_vals 
                                            <= n_IQRu[b,mi]).tolist()

                    # Fill in the IQRs
                    plt.fill_between(rho_vals, cond_PDF[b,:,mi], 0, where = a_mask, 
                                     alpha=0.5, color=plot_color)
                    plt.fill_between(rho_vals, cond_PDF[b,:,mi], 0, where = n_mask, 
                                     alpha=0.5, color=plot_color)

                if pms.plot_untransformed_PDF:
                    # Plot the untransformed PDF
                    plt.plot(rho_vals, cond_PDF_nt[b,:,mi], color=plot_color, 
                            linestyle="--", label=rf"__nolabel__")

                    if pms.plot_statistics:
                        # Plot the analytic mode, untransformed
                        plt.plot(a_mode_no_transform[b,mi], 
                            ddfunc.dn(a_mode_no_transform[b,mi], m, beta_vals[b],
                            transform_pdf=False) / norm_nt[b,:,mi], 
                            'o', color='red', label='__nolabel__')

                        # Plot the numeric untransformed mode
                        plt.plot(n_mode_nt[b,mi], 
                            ddfunc.dn(n_mode_nt[b,mi], m, beta_vals[b], 
                            transform_pdf=False) / norm_nt[b,:,mi],
                            "*", color="blue", label='__nolabel__')
                
                if pms.verbose:
                    print(f"Plot finished for mass = {m:.2E}")
            
            # Finish plot of PDF slices
            plt.xlabel(r"$\tilde{\rho}$")
            plt.ylabel(r"$P_n$")
            plt.title(r"PDF slices along mass")
            plt.grid(True)
            plt.legend()
            plt.savefig("plots/joint-pdf-slice.pdf")
            plt.close()

        else:
            print("Starting most probable profile vs. mass plot...")
            for gi, g in enumerate(gamma_slices):
                # Numerically construct the conditional PDF
                cond_PDF, norm = ddc.calc_PDF(True, g=g)
                
                # Calculate the mode
                n_modes, n_stdevs = ddfunc.n_modes_variances(cond_PDF, rho_vals)

                a_modes = ddfunc.most_probable_rho_transformed(MS[:,0,:], 
                                                    BTS[:,0,:], gamma=g)

                # Calculate the other statistics
                n_medians, n_IQRl, n_IQRu = ddfunc.n_quantiles(cond_PDF, rho_vals)

                a_stats = []
                guesses = np.array([n_modes, n_modes - n_stdevs, n_modes + n_stdevs])
                for i, s in np.ndenumerate(np.array([0.5, pms.lqr, pms.uqr])):
                    nm = NewtonsMethod(MS[:,0,:], BTS[:,0,:], guesses[i], g, s)
                    nm.run()
                    a_stats.append(nm.return_solution())

                # nm = NewtonsMethod(MS[:,0,:], BTS[:,0,:], n_modes, g, 0.5)
                # nm.run()
                # a_median = nm.return_solution()
                
                # Plot analytic and numeric mode
                line, = plt.plot(beta_vals, a_modes[:, gi], 
                                label=rf"m={mass_vals[gi]:.2E}, $\gamma$={g:.2E}")
                mass_color = line.get_color()
                line, = plt.plot(beta_vals, n_modes[:, gi], 
                                color=mass_color, linestyle="--")

                # Plot analytic and numeric median
                plt.plot(beta_vals, a_stats[0][:, gi], linestyle="-.", color=mass_color)
                plt.plot(beta_vals, n_medians[:, gi], label="__nolabel__", 
                         color=mass_color, linestyle="dotted")

                # Plot analytic and numeric IQR
                plt.fill_between(beta_vals, a_stats[1][:, gi], a_stats[2][:, gi], 
                                 alpha=0.5, label="analytic IQR")
                plt.fill_between(beta_vals, n_IQRl[:, gi], n_IQRu[:, gi], 
                                 alpha=0.3, label="numeric IQR")

            plt.xlabel(r"$\beta$")
            plt.ylabel(r"$\hat{\rho}$")
            plt.title(r"Most probale profile vs. $\beta$")
            plt.grid(True)
            plt.legend()
            plt.savefig("plots/mpp-beta-scaling-with-iqrs.pdf")
            plt.close()


        if pms.plot_rho_derivative:
            def rho_derivative(rho):
                delta_c = ddfunc.delta_c_0(1) * func.D(1) / func.D(1)
                return pow(rho, (-1 - 1/delta_c))

            # Plot rho_derivative using rho_vals
            y_vals = [rho_derivative(r) for r in rho_vals]
            plt.figure()
            plt.plot(rho_vals, y_vals)
            plt.xlabel(r"$\bar{\rho}$")
            plt.ylabel(r"$d \tilde{\delta}_l / d \bar{\rho}$")
            plt.grid(True, which='both', ls='--', alpha=0.3)
            plt.title(r"rho derivative: $\bar{\rho}^{-1/\tilde{\delta}_{c}-1}$")
            func.make_directory("plots")
            plt.savefig("plots/rho-derivative.pdf")
            plt.close()

        