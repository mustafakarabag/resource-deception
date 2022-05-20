import os

from markov_decision_processes.mdp import MDP
#from markov_decision_processes.averagemdp import AverageMDP
from markov_decision_processes.totalcostmdp import TotalCostMDP
from markov_decision_processes.timeproductmdp import TimeProductMDP
from grid_world.gridworld import GridWorld
from zero_sum_game.zero_sum import ZeroSumGame
from deception_methods.deceptionmethods import DeceptionMethods
import numpy as np

import pickle

def mymain():

    # Simulation paramaters
    num_of_rows = 10
    num_of_cols = 10
    num_of_states = num_of_rows*num_of_cols
    obstacles = [np.array([2,1]), np.array([2,2]), np.array([3,1]), np.array([3,2]),
                 np.array([3,6]), np.array([3,7]), np.array([4,6]), np.array([4,7]),
                 np.array([6,3]), np.array([6,4]), np.array([6,5]),
                 np.array([7,3]), np.array([7,4]), np.array([7,5])]
    slip_probability = 0
    time_horizon = 8
    init_state_dist = np.asarray([0.0]*num_of_states); init_state_dist[3] = 1 ; #init_state_dist[9] =0.5
    reward_constant = -10
    end_states = [num_of_states-num_of_cols, num_of_states-num_of_cols + 4, num_of_states-1]
    regularization_constant = 4
    num_of_utility_matrices = 2
    true_matrix_index = 0 # True utility matrix is the first utility matrix
    utility_matrices = []

    utility_matrices.append(np.fliplr(np.diag([-1,2,6])))
    utility_matrices.append(np.fliplr(np.diag([6,2,-1])))

    #utility_matrices.append(np.array([[1,0,3], [0,1,3], [0,0,3]]))
    #utility_matrices.append(np.array([[0,1,4], [0,0,1], [-1,1,0]]))
    print(utility_matrices)


    # Create the grid world
    my_grid_world = GridWorld(num_of_rows, num_of_cols, init_state_dist, obstacles, slip_probability, reward_constant, end_states)


    """
    Prediction Algorithm: Solve the entropy-regularized cost minimization problem to compute
                          adversary predictions for each potential utility matrix
    """
    # For each utility matrix:
    # Compute "expected" policies using the entropy-regularized prediction algorithm
    expected_policies = []
    for matrix_index in range(num_of_utility_matrices):
        print("---------------------")
        print(str(matrix_index) + " :Processing predictions for the potential utility matrix:")

        # Solve the optimization problem for prediction
        utility_matrix = utility_matrices[matrix_index]
        _, _, strategy = ZeroSumGame.compute_equilibrium(utility_matrix)
        print(strategy)
        _, _, adversary_strategy = ZeroSumGame.compute_equilibrium_for_column_player(utility_matrix)
        final_distribution = [end_states, np.round(strategy[0,:],3)]
        adversary_distribution = [end_states, np.round(adversary_strategy[:,0],3)]
        if matrix_index == 0:
            final_distribution = [end_states, np.array([1, 0, 0])]
        else:
            final_distribution = [end_states, np.array([0, 0, 1])]
        print("Final distribution is: " + str(final_distribution))
        print("Adversary distribution is: " + str(adversary_distribution))
        val, _,  x_sa = TotalCostMDP.maximize_reward_with_concave_regularizer(my_grid_world, end_states,
                                                                              'entropy', regularization_constant,
                                                                              my_grid_world.initial_state_dist, final_distribution)
        # Extract the corresponding policy
        x_sa[x_sa <= 0] = 0
        policy = []
        x_sa_ind = 0
        for state in range(my_grid_world.NS):
            num_of_acts = my_grid_world.NA_list[0, state]
            if sum(x_sa[0,x_sa_ind:(x_sa_ind+num_of_acts)]) > 0:
                distribution = x_sa[0,x_sa_ind:(x_sa_ind+num_of_acts)] / sum(x_sa[0,x_sa_ind:(x_sa_ind+num_of_acts)])
            else:
                distribution = num_of_acts * [1/num_of_acts]
            policy.append(list(np.round(distribution,3)))
            x_sa_ind = x_sa_ind + num_of_acts


        # Save the occupancy measures for illustrations
        save_file_str = os.path.join(os.path.abspath(os.path.curdir), 'logs', 'prediction_' + str(matrix_index) + '_results', '3stepcontinueMDP.pkl')
        with open(save_file_str, 'wb') as f:
            save_dict = {}
            save_dict['mdp'] = my_grid_world
            save_dict['x_sa'] = x_sa
            save_dict['obstacles'] = obstacles
            save_dict['end_states'] = end_states
            pickle.dump(save_dict, f)

        expected_policies.append(policy)


    """
    Planning Algorithm: Solve the cost minimization problem to compute
                        the deception policies for exaggeration and ambiguity
    """

    print("---------------------")
    print("Processing the planning phase:")

    # Compute the expanded MDP and corresponding target states and distributions
    time_product_grid_world = TimeProductMDP(my_grid_world, time_horizon, 'continue')
    product_end_states = list(time_horizon*num_of_states+np.array(end_states))
    print(product_end_states)
    _, _, strategy = ZeroSumGame.compute_equilibrium(utility_matrices[true_matrix_index])
    product_final_distribution = [product_end_states, np.round(strategy[0,:],3)]
    product_final_distribution = [product_end_states, np.array([1,0,0])]


    reward_list_transient = []
    reward_list_steady_state = []
    eps = 1e-5
    matrix_index = 1
    for state in range(num_of_states):
        act_rewards = np.log((np.array(expected_policies[matrix_index][state])+eps)/(np.array(expected_policies[true_matrix_index][state])+eps))
        reward_list_transient.append(np.array([act_rewards]))
        reward_list_steady_state.append(-0.1*np.ones((1,len(act_rewards))))

    reward_list = reward_list_transient * time_horizon + reward_list_steady_state

    # Compute deceptive policy for exaggeration behavior
    #val, _,  x_sa_2 = TotalCostMDP.maximize_reward_with_concave_regularizer(time_product_grid_world, product_end_states,
    #                                                                        'none', regularization_constant,
    #                                                                        time_product_grid_world.initial_state_dist,
    #                                                                        product_final_distribution,
    #                                                                        reward_list)

    val, x_sa, max_index = DeceptionMethods.exaggration(time_product_grid_world,expected_policies,true_matrix_index,product_end_states,0.1,product_final_distribution)

    #val, x_sa = DeceptionMethods.ambiguity(time_product_grid_world,expected_policies,true_matrix_index,product_end_states,0.1,product_final_distribution)

    save_file_str = os.path.join(os.path.abspath(os.path.curdir), 'logs', 'deception_results', '3stepcontinueMDP.pkl')
    with open(save_file_str, 'wb') as f:
        save_dict = {}
        save_dict['time_product_mdp'] = time_product_grid_world
        save_dict['x_sa'] = x_sa
        save_dict['obstacles'] = obstacles
        save_dict['end_states'] = end_states
        pickle.dump(save_dict, f)



if __name__ == '__main__':
    mymain()
