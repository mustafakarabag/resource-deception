import numpy as np

from markov_decision_processes.mdp import MDP
import cvxpy as cp


class TotalCostMDP:
    """
    Class for functions related to MDPs with infinite horizons and total statistics.

    The states that are not in end states are assumed to be transient.
    """

    def compute_occupancy_measures(mdp: MDP, policy) -> np.ndarray:
        """
        Computes the average occupancy measures of given an MDP and a policy.
        Refer to Puterman(1994) for relevant technical details.

        :param mdp: A discrete time, discrete state MDP.
        :param policy: A stationary (potentially randomized) policy.
        :return: A vector of occupancy measures that add up to 1. The occupancy measures are stacked first wrt the state
        indices and then wrt action indices.
        """

        #Check the validity of the policy for the given policy
        assert mdp.is_policy_valid(policy)

        x_sa = cp.Variable((1, mdp.NSA))
        occupancy_constraints, x_s_inflow = TotalCostMDP._create_flow_equations(mdp, x_sa)

        #Problem constraints
        constraints = []
        constraints.extend(occupancy_constraints)

        # Policy probabilities are followed
        policy_constraints = []
        sa_ind = 0
        for state_index in range(mdp.NS):
            num_of_actions = int(mdp.NA_list[0, state_index])
            for action_index in range(num_of_actions - 1):
                policy_constraints.append(x_sa[0,sa_ind+action_index]*policy[state_index][0, action_index+1] == x_sa[0,sa_ind+action_index+1]*policy[state_index][0,action_index])
            sa_ind += mdp.NA_list[0, state_index]
        constraints.extend(policy_constraints)

        #Feasibility objective
        constant_one = 1.0
        obj = cp.Maximize(constant_one)

        # Create the feasibility problem
        prob = cp.Problem(obj, constraints)

        prob.solve()
        #print("\nThe optimal value is", prob.value)
        #print("A solution x is")
        #print(x_sa.value)
        return x_sa.value, x_s_inflow.value

    def maximize_reward_with_concave_regularizer(mdp: MDP, end_states, regularizer: str, reg_constant: float, initial_state_dist, final_state_dist=None):
        """
        Computes the policy that collects maximum average reward subject to regularization.
        This implementation assumes that every strongly connected component is reachable with probability 1.
        To compute maximum reward policy, set regularizer to 'none'.

        :param mdp: A discrete time, discrete state MDP.
        :param reward: A list of reward arrays for each state.
        :param end_states: A list of end states.
        :param regularizer: The type of regularization. Potential values are 'entropy', 'none'
        :param reg_constant: Regularization constant that multiplies the regularization term.
        :param initial_state_dist: A distribution over initial states.
        :param final_state_dist: A tuple with a support over some states and respective probabilities.
        :return: The optimal objective value, the value for the linear reward w/o regularization, the occupancy measures
        """
        #TODO add KL and Fisher

        ##Check the validity of the list of rewards
        #assert mdp.is_reward_function_valid(reward)
        reward = mdp.reward

        reward_vec = TotalCostMDP.state_list_to_stateaction_vec(mdp, reward)

        #Tolerance
        tol = 1e-7

        #Occupancy measure
        x_sa = cp.Variable((1, mdp.NSA))

        #Occupancy constraints that an average MDP obeys
        occupancy_constraints, x_s_inflow = TotalCostMDP._create_flow_equations_zero_unreachable(mdp, x_sa, end_states, initial_state_dist)

        #Problem constraints
        constraints = []
        constraints.extend(occupancy_constraints)

        final_state_constraints = []
        #Final state constraints
        if final_state_dist is not None:
            (target_states, target_probabilities) = final_state_dist
            for i in range(len(target_states)):
                final_state_constraints.append(x_s_inflow[target_states[i]] == target_probabilities[i])
        constraints.extend(final_state_constraints)

        linear_reward_func = x_sa @ reward_vec
        reg_func = 0.0
        if(regularizer == 'entropy'):
            sa_ind = 0
            for state_index in range(mdp.NS):
                x_sa_current_state = x_sa[:, sa_ind:(sa_ind + mdp.NA_list[0, state_index])]
                reg_term_state = -cp.sum(cp.rel_entr(x_sa_current_state, cp.sum(x_sa_current_state)*np.ones(x_sa_current_state.shape)))
                reg_func += reg_constant*reg_term_state
                sa_ind += mdp.NA_list[0, state_index]
        elif(regularizer == 'none'):
            print('No regularizer')
        else:
            print('Unrecognized regularizer ignoring it')

        func = linear_reward_func + reg_func
        obj = cp.Maximize(func)


        # Create the problem
        prob = cp.Problem(obj, constraints)

        prob.solve()
        print("\nThe optimal value is", prob.value)
        print("The reward_function is ", x_sa.value @ reward_vec)
        #print("A solution x is")
        #print(x_sa.value)
        return prob.value, x_sa.value @ reward_vec,  x_sa.value



    def _create_flow_equations(mdp: MDP, x_sa, end_states, initial_state_dist):
        """
        Creates the flow equations that are required for policy comnputations.

        :param mdp: A discrete time, discrete state MDP.
        :param x_sa: A vector of cvxpy variables for occupancy variables.
        :param end_states: A list of absorbing states.
        :param initial_state_dist: A numpy array of the initial state distribution
        :return: A list of affine constraints that ensure the validity of the policy to be computed.
        """
        x_s_inflow = [None] * mdp.NS
        x_s_outflow = [None] * mdp.NS

        sa_ind = 0
        for state_index in range(mdp.NS):
            if state_index not in end_states:
                #Outflow is the total occupancy measure of the actions
                x_s_outflow[state_index] = cp.sum(x_sa[:, sa_ind:(sa_ind + mdp.NA_list[0, state_index])])
                (succ_state_list, tran_mat) = mdp.list_of_states_and_transitions[state_index]
                num_of_actions = mdp.NA_list[0, state_index]
                outflows = x_sa[:, sa_ind:(sa_ind + num_of_actions)] @ tran_mat
                for succ_state_index in range(succ_state_list.size):
                    succ_state = succ_state_list[0,succ_state_index]
                    #Distribute outflow to the successor states
                    if x_s_inflow[succ_state] is None:
                        x_s_inflow[succ_state] = outflows[0, succ_state_index]
                    else:
                        x_s_inflow[succ_state] = cp.atoms.affine.add_expr.AddExpression([x_s_inflow[succ_state], outflows[0, succ_state_index]])
                sa_ind += mdp.NA_list[0, state_index]

                #Add initial flows
                x_s_inflow[state_index] = x_s_inflow[state_index] + initial_state_dist[state_index]

        # Define the occupancy measure constraints
        occupancy_constraints = []

        # All occupancy measures are positive
        occupancy_constraints.append(x_sa >= 0.0)

        # Incoming flow is equal to the outgoing flow
        for state_index in range(mdp.NS):
            if state_index not in end_states:
                occupancy_constraints.append(x_s_inflow[state_index] == x_s_outflow[state_index])

        return occupancy_constraints, x_s_inflow

    def _create_flow_equations_zero_unreachable(mdp: MDP, x_sa, end_states, initial_state_dist):
        """
        Creates the flow equations that are required for policy comnputations. Sets zero occupancy for
        the states that are unreachable.

        :param mdp: A discrete time, discrete state MDP.
        :param x_sa: A vector of cvxpy variables for occupancy variables.
        :param end_states: A list of end states.
        :return: A list of affine constraints that ensure the validity of the policy to be computed.
        """
        occupancy_constraints = TotalCostMDP._create_flow_equations(mdp, x_sa, end_states, initial_state_dist)
        reachable_states = mdp.find_reachable_states()
        sa_ind = 0
        for state_index in range(mdp.NS):
            if state_index not in reachable_states:
                occupancy_constraints.append(x_sa[:, sa_ind:(sa_ind + mdp.NA_list[0, state_index])] == 0)
            sa_ind += mdp.NA_list[0, state_index]
        return occupancy_constraints

    def state_list_to_stateaction_vec(mdp: MDP, var_list: list):
        """
        Creates a vector of variables that are given in a list.

        :param mdp: A discrete time, discrete state MDP.
        :param var_list: A list whose i-th entry contains a vector of variables for State i.
        :return: A vector of variables that are stacked first wrt the state indices and then wrt to action indices.
        """
        var_vec = np.zeros((mdp.NSA, 1))
        sa_ind = 0
        for state_index in range(mdp.NS):
            var_vec[sa_ind:(sa_ind + mdp.NA_list[0, state_index]), 0] = var_list[state_index]
            sa_ind += mdp.NA_list[0, state_index]
        return var_vec
