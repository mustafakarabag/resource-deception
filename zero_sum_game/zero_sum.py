import numpy as np
import cvxpy as cp


class ZeroSumGame:
    """
    Class for functions related to zero sum games.
    """
    def compute_equilibrium(utility_matrix):

        """
        @param utility_matrix: NxN np.array with non-negative elements that describes the utilities in the game
        """
        support_size_for_strategy = np.size(utility_matrix,0)

        # Tolerance level
        tol = 1e-7

        # Variables
        strategy_distribution = cp.Variable((1, support_size_for_strategy))
        security_level = cp.Variable((1,1))

        # Constraints
        constraints = []
        constraints.append(strategy_distribution @ utility_matrix - security_level @ np.ones((1, support_size_for_strategy)) >= 0)
        constraints.append(cp.sum(strategy_distribution) == 1)
        constraints.append(strategy_distribution >= 0)

        obj = cp.Maximize(security_level)


        # Create the problem
        prob = cp.Problem(obj, constraints)

        prob.solve()
        print("\nThe optimal value is", prob.value)

        return prob.value, security_level.value, strategy_distribution.value
