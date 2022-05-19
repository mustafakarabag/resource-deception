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
        print("The security level is ", prob.value)

        return prob.value, security_level.value, strategy_distribution.value

    def compute_equilibrium_for_column_player(utility_matrix):

        """
        @param utility_matrix: NxN np.array with non-negative elements that describes the utilities in the game
        """
        support_size_for_strategy = np.size(utility_matrix,1)
        # Tolerance level
        tol = 1e-7

        # Variables
        strategy_distribution = cp.Variable((support_size_for_strategy,1))
        security_level = cp.Variable((1,1))

        # Constraints
        constraints = []
        constraints.append(utility_matrix @ strategy_distribution -  np.ones((support_size_for_strategy, 1)) @ security_level <= 0)
        constraints.append(cp.sum(strategy_distribution) == 1)
        constraints.append(strategy_distribution >= 0)

        obj = cp.Minimize(security_level)


        # Create the problem
        prob = cp.Problem(obj, constraints)

        prob.solve()
        print("The security level is ", prob.value)
        return prob.value, security_level.value, strategy_distribution.value
