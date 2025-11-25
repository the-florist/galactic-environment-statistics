"""
    Filename: double_distribution_calculations.py
    Author: Ericka Florio
    Created: 21 Nov. 2025
    Description: Declaration of a class that 
            calculates the joint double distribution for the number density of objects 
            of mass m with local overdensity delta_l, 
            as derived in Pavlidou and Fields 2005.
"""

import numpy as np
import util.parameters as pms
import util.functions as func
import util.double_distribution_functions as ddfunc
from util.Newton_method import NewtonsMethod

class DoubleDistributionCalculations:
    def __init__(self):
        self.bvs = np.linspace(pms.beta_min, pms.beta_max, pms.num_beta)
        self.rvs = np.linspace(pms.rho_tilde_min, pms.rho_tilde_max, pms.num_rho)
        self.mvs = np.linspace(pms.mass_min, pms.mass_max, pms.num_mass) # np.array([pms.M_200]) # np.array([1.3, 5, 17]) * 1e14 # np.linspace(pms.mass_min, pms.mass_max, pms.num_mass)
        self.gamma_slices = np.linspace(pms.gamma_min, pms.gamma_max, pms.num_gamma)

        self.BTS, self.RHOS, self.MS = np.meshgrid(self.bvs, self.rvs, 
                                                   self.mvs, indexing='ij')

    
    def calc_PDF(self, transfm, g = pms.default_gamma, sis = False):
        if pms.verbose:
            print("Starting PDF calculation.")

        self.PDF = ddfunc.dn(self.RHOS, self.MS, self.BTS, 
                             transform_pdf=transfm, gamma=g, 
                             sis=sis)
        
        if pms.normalise_pdf:
            if pms.plot_dimension == 2:
                axs = (1, 2)
            elif pms.plot_dimension == 1:
                axs = 1
            
            self.norm = self.PDF.sum(axis=axs, keepdims=True)
            self.PDF /= self.norm
            if pms.verbose:
                print(f"PDF norm - 1: {abs(self.PDF.sum(axis=axs) - 1.).max()}")
        

    def find_quantile(self, zscore):
        sm = 0
        # Track the CDF, and iqrs found
        stat = np.zeros_like(self.PDF[:,0,:])

        # Use the numerical CDF to find the iqrs
        for idx, x in enumerate(self.rvs):
            sm += self.PDF[:,idx,:]
            sm_l = np.argwhere(sm > zscore)
            if sm_l.size != 0:
                for i in range(len(sm_l)):
                    f = sm_l[i, 0]
                    s = sm_l[i, 1]
                    if stat[f, s] == 0:
                        stat[f, s] = x
        return stat
    
    
    def n_stats(self):
        """
            Calculate the mode of the double distribution 
            (i.e. the most probable profile)
            and the standard deviation of the mode, sliced at m.
        """

        try:
            self.n_mode = self.rvs[np.argmax(self.PDF, axis=1)]
            # for this variance measure, multiply PDF with 
            # squared difference from the sample mode.
            self.n_stdev = np.array([[(self.PDF[j,:,i] * pow(self.rvs 
                                                - self.n_mode[j, i], 2)).sum() 
                                                for i in range(len(self.n_mode[0]))] 
                                                for j in range(len(self.n_mode))])
            self.n_stdev /= (pms.num_rho - 1)
        
        except:
            self.n_mode = self.rvs[np.argmax(self.PDF)]
            self.n_stdev = np.array([(self.PDF[i] * pow(self.rvs 
                                                - self.n_mode, 2)).sum() 
                                                for i in range(len(self.PDF))])
            self.n_stdev /= (pms.num_rho - 1)

        temp = [self.find_quantile(0.5), self.find_quantile(pms.lqr), 
                                                     self.find_quantile(pms.uqr)]
        self.n_quantiles = np.stack(temp, axis=0)

    
    
    def a_stats(self, transfm, g = pms.default_gamma):
        
        if transfm:
            self.a_mode = ddfunc.most_probable_rho_transformed(self.MS[:,0,:], 
                                                        self.BTS[:,0,:], gamma=g)
            temp = []
            guesses = np.array([self.n_mode, self.n_mode - self.n_stdev, 
                                            self.n_mode + self.n_stdev])
            
            for i, s in np.ndenumerate(np.array([0.5, pms.lqr, pms.uqr])):
                nm = NewtonsMethod(self.MS[:,0,:], self.BTS[:,0,:], guesses[i], g, s)
                nm.run()
                temp.append(nm.return_solution())
            
            self.a_quantiles = np.stack(temp, axis=0)
        
        else:
            self.a_mode = ddfunc.most_probable_rho(self.MS[:,0,:], self.BTS[:,0,:], 
                                                   inc_mass_scaling=True)

    
    def print_stats_comparison(self, mi, bi):
        n_median = self.find_quantile(0.5)[bi, mi]
        a_median = self.a_quantiles[1, bi, mi]
        nm = NewtonsMethod(self.mvs, self.bvs, 0)

        n_cdf = ddfunc.conditional_CDF(n_median, self.mvs[mi], self.bvs[bi])
        a_cdf = ddfunc.conditional_CDF(a_median, self.mvs[mi], self.bvs[bi])
        
        print("Median estimates: ", n_median, a_median)
        print("Conditional CDF of each: ", n_cdf, a_cdf)
        
        print("Target fn for each: ")
        print(nm.target_fn(n_median)[bi, mi])
        print(nm.target_fn(a_median)[bi, mi])


    def rho_derivative(self, rho):
            delta_c = ddfunc.delta_c_0(1) * func.D(1) / func.D(1)
            return pow(rho, (-1 - 1/delta_c))

    
    def calc_mode_error(self):
        transf_mode = ddfunc.most_probable_rho_transformed(self.MS[:,0,:], 
                                        self.BTS[:,0,:], pms.default_gamma)

        us_mode = ddfunc.most_probable_rho(self.MS[:,0,:], self.BTS[:,0,:], 
                                                   inc_mass_scaling=True)

        num_mode = self.rvs[np.argmax(self.PDF, axis=1)]
        
        self.us_transf_diff = (us_mode - transf_mode) / us_mode
        self.us_num_diff = (us_mode - num_mode) / us_mode

        # return us_transf_diff, us_num_diff