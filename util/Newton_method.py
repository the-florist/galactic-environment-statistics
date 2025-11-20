"""

"""
import numpy as np
import util.parameters as pms
import util.double_distribution_functions as ddfunc

class NewtonsMethod:
    # Rho domain parameters
    x_max = pms.rho_tilde_max
    x_min = pms.rho_tilde_min
    initial_step = (pms.rho_tilde_max - pms.rho_tilde_min) / pms.num_rho

    # Newton's method iteration parameters
    max_iterations = 100
    tol = pms.root_finder_precision

    def __init__(self, masses, betas, guess, gamma = pms.default_gamma, 
                                             score = 0.5):
        """
            Initialise mass, beta and initial guess arrays,
            and gamma and zscore parameters.
        """
        # Iterate over a smaller param space, used for debugging
        # mi = 3
        # bi = 15

        self.ms = masses
        self.bs = betas
        self.gamma = gamma
        self.guess = guess
        self.zscore = score
        self.solution = np.zeros_like(betas)

    def return_solution(self):
        return self.solution

    def target_fn(self, rho):
        """
            Calculate the difference between the CDF and the zscore 
            at a value of rho.
        """
        return ddfunc.conditional_CDF(rho, self.ms, self.bs, 
                                         self.gamma, pms.a_f) - self.zscore

    def deriv(self, rho_0, rho_1, step):
        """
            Calculate the first derivative between two values of rho.
        """
        return (self.target_fn(rho_1) - self.target_fn(rho_0)) / step

    def run(self):
        """
            Perform the Newton's method iteration, storing converged values 
            in the solutions array, and print the max error across the 
            parameter space at the end.
        """

        # Init the iterator and mask arrays
        it = 0
        mask = np.full(self.bs.shape, False)

        # Set the first two steps according to the guess
        x0 = self.guess
        x1 = x0 - self.initial_step

        print("Starting Newton's method loop...")
        while it < self.max_iterations:
            # Find the current step size
            dx = (x1 - x0)

            # Confirm x1 != 0, before plugging into target fn
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

            # Calculate and the next step and apply floor/cieling
            temp = x1 - self.target_fn(x1) / d
            temp[temp > pms.rho_tilde_max] = pms.rho_tilde_max
            temp[temp < pms.rho_tilde_min] = pms.rho_tilde_min

            # See if any of the parameter points have converged
            # If they have, save in solutions array and update mask
            index_1 = np.argwhere(abs(temp - x1) < self.tol)
            if index_1.size != 0:
                for i in index_1:
                    i1, i2 = i
                    if mask[i1, i2] == False:
                        self.solution[i1, i2] = temp[i1, i2]
                        mask[i1, i2] = True
                    else:
                        continue

            # Check end/break conditions, or iterate
            if np.all(mask == True):
                print("Convergence complete.")
                break
            elif it == (self.max_iterations - 1):
                print("NewtonsMethod:run, failed to converge after "+
                      f"{self.max_iterations} steps.")
                exit()
            else:
                it += 1
                for i, m in np.ndenumerate(mask):
                    x, y = i
                    if m == False:
                        x0[x, y] = x1[x, y]
                        x1[x, y] = temp[x, y]
                    else:
                        continue

        print("Max error on root: ", self.target_fn(self.solution).max())