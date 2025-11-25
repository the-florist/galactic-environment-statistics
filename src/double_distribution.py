"""
    Filename: double_distribution.py
    Author: Ericka Florio
    Created: 11 Sept 2025
    Description: Main class which controls the calculation of 
                 the double distribution and the plotting of this 
                 distribution, as described in 
                 double_distribution_calculations.py and 
                 double_distribution_plotting.py.
"""

import util.parameters as pms
from src.double_distribution_calculations import DoubleDistributionCalculations
from src.double_distribution_plotting import DoubleDistributionPlots

class DoubleDistribution():
    def __init__(self):
        self.plot_dim = pms.plot_dimension
        self.slice_in_rho = pms.slice_in_rho
        self.slice_in_beta = pms.slice_in_beta
        self.rho_deriv = pms.plot_rho_derivative
        self.plot_mode_error = pms.mode_error

        self.ddc = DoubleDistributionCalculations()
        self.ddp = DoubleDistributionPlots(self.ddc)

    def joint_PDF(self):
        """
            Calculate and plot the normalised joint PDF.
        """
        self.ddc.calc_PDF(True, pms.default_gamma)
        self.ddp.plot_heatmap("plots/joint-pdf.pdf")

    def pdf_slice_in_rho(self):
        """
            Calculate the conditional PDF at slices of rho or beta, and 
            plot alongside the mode, median and quantiles.
        """
        
        # Find the closest beta to our heuristic value
        transform_pdf = True

        # Construct the conditional PDF and its statistics
        self.ddc.calc_PDF(transform_pdf, pms.default_gamma)
        self.ddc.n_stats()
        self.ddc.a_stats(transform_pdf)

        # Plot the full PDF
        for mi in range(len(self.ddc.mvs)):
            self.ddp.plot_rho_slice(mi, transform_pdf)
            if pms.plot_statistics:
                self.ddp.plot_a_stats(mi, transform_pdf)

        # Construct and plot the untransformed PDF and its statistics
        if pms.plot_untransformed_PDF:
            transform_pdf = False
            self.ddc.calc_PDF(transform_pdf, pms.default_gamma)
            self.ddc.n_stats()
            self.ddc.a_stats(transform_pdf)

            for mi in range(len(self.ddc.mvs)):
                self.ddp.plot_rho_slice(mi, transform_pdf)
                if pms.plot_statistics:
                    self.ddp.plot_a_stats(mi, transform_pdf)

        # Finish plot of PDF slices
        self.ddp.format_plot(r"PDF slices along mass", r"$\tilde{\rho}$", r"$P_n$")
        self.ddp.save_plot("plots/joint-pdf-slice.pdf")

    def pdf_slice_in_beta(self):
        """
            Plot the conditional PDF with respect to beta, for a range of 
            mass/gamma values (replicates Fig. 3 of 
            Korkidis and Pavlidou 2024).
        """

        for gi, g in enumerate(self.ddc.gamma_slices):
            # Numerically construct the conditional PDF
            transform_pdf = True
            self.ddc.calc_PDF(transform_pdf, g=g)
            
            # Calculate the numerical statistics
            self.ddc.n_stats()
            self.ddc.a_stats(transform_pdf, g=g)

            # Plot the stats as a function of beta
            self.ddp.plot_beta_slices(gi)

        # Format and save the plot
        self.ddp.format_plot(r"Most probale profile vs. $\beta$", 
                        r"$\beta$", r"$\hat{\rho}$")
        self.ddp.save_plot("plots/mpp-scaling.pdf")

    def mode_error(self):
        # Find the closest beta to our heuristic value
        transform_pdf = True

        # Construct the conditional PDF and its statistics
        self.ddc.calc_PDF(transform_pdf, pms.default_gamma)
        self.ddc.calc_mode_error()
        self.ddp.plot_mode_error()


    def run(self):
        """
            Perform the requested calculation/plotting routine based on the 
            parameters laid out in parameters.py.
        """

        if pms.verbose:
            print(f"Starting {pms.plot_dimension}D plot generation...")

        if self.plot_dim == 2:
            self.joint_PDF()

        elif self.plot_dim == 1 and self.slice_in_rho:
            self.pdf_slice_in_rho()

        elif self.plot_dim == 1 and self.slice_in_beta:
            self.pdf_slice_in_beta()

        elif self.plot_dim == 1 and self.plot_mode_error:
            self.mode_error()

        elif self.rho_deriv:
            self.ddp.plot_rho_derivative()

        else:
            print(pms.plot_dimension, pms.slice_in_rho, 
                  pms.slice_in_beta, pms.plot_rho_derivative)
            raise ValueError("DoubleDistribution:run : plot configuration is "
                             "incorrect or not implemented.")


            