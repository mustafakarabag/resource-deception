from markov_decision_processes.mdp import MDP
from markov_decision_processes.timeproductmdp import TimeProductMDP
from markov_decision_processes.totalcostmdp import TotalCostMDP
import cvxpy as cp
import numpy as np
import math

class DeceptionMethods():
    def exaggration(time_product_mdp:TimeProductMDP, expected_policies, true_matrix_index, product_end_states, transient_step_cost, product_final_dist=None):
        x_sa_cvx = cp.Variable((1, time_product_mdp.NSA))
        num_of_org_states = time_product_mdp.original_mdp.NS
        time_horizon = time_product_mdp.T
        reward_list_transient = []
        reward_list_steady_state = []
        eps = 1e-5

        max_val = -math.inf
        max_index = true_matrix_index
        max_val_x_sa = 0
        for matrix_index in range(len(expected_policies)):
            if matrix_index != true_matrix_index:
                for state in range(num_of_org_states):
                    act_rewards = np.log((np.array(expected_policies[matrix_index][state])+eps)/(np.array(expected_policies[true_matrix_index][state])+eps))
                    reward_list_transient.append(np.array([act_rewards]))
                    reward_list_steady_state.append(-transient_step_cost*np.ones((1,len(act_rewards))))
                reward_list = reward_list_transient * time_horizon + reward_list_steady_state
                obj_func = TotalCostMDP.create_linear_objective(time_product_mdp, x_sa_cvx, reward_list)

                # Compute deceptive policy for exaggeration behavior
                val, x_sa = TotalCostMDP.maximize_given_objective(time_product_mdp, x_sa_cvx, obj_func, [], product_end_states, time_product_mdp.initial_state_dist, product_final_dist)

                if val > max_val:
                    max_val = val
                    max_index = matrix_index
                    max_val_x_sa = x_sa

        return max_val, max_val_x_sa, max_index

    def ambiguity(time_product_mdp:TimeProductMDP, expected_policies, true_matrix_index, product_end_states, transient_step_cost, product_final_dist=None):
        x_sa_cvx = cp.Variable((1, time_product_mdp.NSA))
        num_of_org_states = time_product_mdp.original_mdp.NS
        time_horizon = time_product_mdp.T
        reward_list_transient = []
        reward_list_steady_state = []
        eps = 1e-5

        for state in range(num_of_org_states):
            act_rewards = np.zeros(np.array(expected_policies[0][state]).shape)
            reward_list_transient.append(np.array([act_rewards]))
            reward_list_steady_state.append(-transient_step_cost * np.ones((1, len(act_rewards))))
        reward_list = reward_list_transient * time_horizon + reward_list_steady_state
        obj_func_lin = TotalCostMDP.create_linear_objective(time_product_mdp, x_sa_cvx, reward_list)

        constraints = []
        max_kl_div = cp.Variable((1,len(expected_policies)))
        states_to_be_included = range(num_of_org_states*time_horizon)
        for matrix_index in range(len(expected_policies)):
            extended_ref_policy = expected_policies[matrix_index]*(time_horizon+1)
            constraints.append(max_kl_div[0,matrix_index] >= TotalCostMDP.create_kl_objective(time_product_mdp, x_sa_cvx, extended_ref_policy,states_to_be_included))
        ambiguity_obj_func = -cp.max(max_kl_div)

        obj_func = obj_func_lin + ambiguity_obj_func
        # Compute deceptive policy for ambiguity behavior
        val, x_sa = TotalCostMDP.maximize_given_objective(time_product_mdp, x_sa_cvx, obj_func, constraints, product_end_states, time_product_mdp.initial_state_dist, product_final_dist)


        return val, x_sa

