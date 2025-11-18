"""

"""
import numpy as np
import matplotlib.pyplot as plt
import util.parameters as pms
import src.double_distribution_functions as ddfunc

class NewtonsMethod:
    max_iterations = 50
    step = (pms.rho_tilde_max - pms.rho_tilde_min) / pms.num_rho
    tol = pms.root_finder_precision
    zscore = 0.5
    x_max = pms.rho_tilde_max
    x_min = pms.rho_tilde_min

    def __init__(self, masses, betas, gamma, guess):
        beg = 0
        end = 3

        self.ms = masses[beg:end,beg:end]
        self.bs = betas[beg:end,beg:end]
        self.gamma = gamma
        self.guess = guess[beg:end,beg:end]

        # print("Masses: ", self.ms)
        # print("Betas: ", self.bs)

    def target_fn(self, x):
        return ddfunc.conditional_CDF(x, self.ms, self.bs, 
                                         self.gamma, pms.a_f) - self.zscore

    def deriv(self, x0, x1, dx):
        dx[dx == 0] = 1
        d = (self.target_fn(x1) - self.target_fn(x0)) / dx
        d[d == 0] = 1
        return d

    def run(self):
        # Init the iterator, step and solution array
        it = 0
        solution = np.zeros_like(self.bs)
        print("Solution shape: ", solution.shape)
        mask = np.full(self.bs.shape, False)

        x0 = self.guess
        x1 = x0 - self.step
        while it < self.max_iterations:
            # Find the current step
            dx = (x1 - x0)
            if np.any(x1 == 0):
                print(x1)
                print("NewtonsMethod:run, invalid x1 encountered.")
                exit()

            # Calculate and check the derivative
            d = self.deriv(x0, x1, dx)
            if np.any(d == 0) or np.isnan(d).any():
                print(d)
                print("NewtonsMethod:run, derivative has returned 0 or nan "+ 
                      f"at step {it}.")
                exit()

            # Calculate and check the next step
            temp = x1 - self.target_fn(x1) / d
                                                      
            if np.any(temp == 0):
                print(temp)
                print("NewtonsMethod:run, iterator temp has left bounds of domain.")
                exit()

            # See if any of the parameter points have converged
            index_1 = np.argwhere(abs(temp - x1) < self.tol)

            if index_1.size != 0:
                for i in index_1:
                    i1, i2 = i
                    if mask[i1, i2] == False:
                        solution[i1, i2] = temp[i1, i2]
                        mask[i1, i2] = True
                    else:
                        continue
            
            if np.all(mask == True):
                print("Convergence complete.")
                break

            x0 = x1
            x1 = temp.copy()
            it += 1
            
        # rho_vals = np.linspace(self.x_min, self.x_max, pms.num_rho)
        # # ds = np.array([self.deriv(rho_vals[i], rho_vals[i+1], self.step) for i in range(len(rho_vals)-1)])
        # func = np.array([self.target_fn(r) for r in rho_vals])
        # plt.plot(rho_vals, func)
        # plt.plot(solution, self.target_fn(solution), '*', color='r')
        # plt.savefig("func.pdf")
        # plt.close()

        print(self.target_fn(solution))