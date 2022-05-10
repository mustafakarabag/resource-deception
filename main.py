import os

from markov_decision_processes.mdp import MDP
#from markov_decision_processes.averagemdp import AverageMDP
from markov_decision_processes.totalcostmdp import TotalCostMDP
from markov_decision_processes.timeproductmdp import TimeProductMDP
from grid_world.gridworld import GridWorld
from zero_sum_game.zero_sum import ZeroSumGame
import numpy as np

import pickle

def mymain():

    # Simulation paramaters
    num_of_rows = 4
    num_of_cols = 4
    num_of_states = num_of_rows*num_of_cols
    obstacles = []
    slip_probability = 1e-5
    time_horizon = 2
    init_state_dist = np.asarray([0.0]*num_of_states); init_state_dist[0] = 1 #; init_state_dist[0] = 0.5
    reward_constant = -10
    end_states = [num_of_states*(time_horizon+1) - num_of_cols, num_of_states*(time_horizon+1)-1]
    utility_matrix = np.random.rand(len(end_states),len(end_states))
    regularization_constant = 0.01

    # Create the grid world and expanded MDP
    my_grid_world = GridWorld(num_of_rows, num_of_cols, init_state_dist, obstacles, slip_probability, reward_constant)
    time_product_grid_world = TimeProductMDP(my_grid_world, time_horizon, 'continue')


    # Compute occupancy measures using the entropy-regularized prediction algorithm
    val, _,  x_sa = TotalCostMDP.maximize_reward_with_concave_regularizer(time_product_grid_world, end_states, 'entropy',
                                                          regularization_constant, time_product_grid_world.initial_state_dist)


    # Save the constructed gridworld
    save_file_str = os.path.join(os.path.abspath(os.path.curdir), 'logs', 'time_product_mdp_results', '3stepcontinueMDP.pkl')
    with open(save_file_str, 'wb') as f:
        save_dict = {}
        save_dict['time_product_mdp'] = time_product_grid_world
        save_dict['x_sa'] = x_sa
        pickle.dump(save_dict, f)



    value, security_level, strategy = ZeroSumGame.compute_equilibrium(utility_matrix)
    final_distribution = [end_states, np.round(strategy[0,:],3)]
    print(final_distribution)


    # Compute deceptive policy for exaggeration behavior
    val, _,  x_sa_2 = TotalCostMDP.maximize_reward_with_concave_regularizer(time_product_grid_world, end_states, 'none',
                                                          regularization_constant, time_product_grid_world.initial_state_dist,
                                                          final_distribution)

    save_file_str = os.path.join(os.path.abspath(os.path.curdir), 'logs', 'deception_results', '3stepcontinueMDP.pkl')
    with open(save_file_str, 'wb') as f:
        save_dict = {}
        save_dict['time_product_mdp'] = time_product_grid_world
        save_dict['x_sa'] = x_sa_2
        pickle.dump(save_dict, f)



if __name__ == '__main__':
    mymain()
