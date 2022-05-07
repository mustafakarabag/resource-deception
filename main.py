import os

from markov_decision_processes.mdp import MDP
#from markov_decision_processes.averagemdp import AverageMDP
from markov_decision_processes.totalcostmdp import TotalCostMDP
from markov_decision_processes.timeproductmdp import TimeProductMDP
from grid_world.gridworld import GridWorld
import numpy as np

import pickle

def mymain():

    # my_mdp = MDP([[np.array([[0,1,2,3]]),np.eye(4)],
    #               [np.array([[0,1,2,3]]),np.eye(4)],
    #                 [np.array([[2]]), np.eye(1)],
    #                 [np.array([[3]]),np.eye(1)]], 0)
    #
    # reward_list = [
    #     np.array([-1.0, -1.0, 0.0, 0.0]),
    #     np.array([-1.0, -1.0, 0.0, 0.0]),
    #     np.array([0.0]),
    #     np.array([0.0])
    # ]
    # end_states = [2,3]
    # final_state_const = ([2,3], np.array([0.2, 0.8]))
    # initial_state_dist = np.array([0.2, 0.8, 0, 0])
    # print(my_mdp)
    # #TotalCostMDP.maximize_reward_with_concave_regularizer(my_mdp, reward_list, end_states, 'entropy', 0.1, initial_state_dist, final_state_const)
    #
    # my_mdp2 = MDP([[np.array([[0,1]]),np.eye(2)],
    #               [np.array([[1,2,3]]),np.eye(3)],
    #                 [np.array([[0,1,2,3]]),np.eye(4)],
    #                 [np.array([[0,1,2]]),np.eye(3)]], 0)
    #
    # time_product_MDP = TimeProductMDP(my_mdp2, 2, 'continue')
    # print(time_product_MDP.product_state_to_original_state(8))
    # print(time_product_MDP.NS)


    num_of_rows = 6
    num_of_cols = 5
    num_of_states = num_of_rows*num_of_cols
    obstacles = []
    slip_probability = 0.2
    init_state_dist = np.asarray([0.0]*num_of_states); init_state_dist[3] = 0.5; init_state_dist[1] = 0.5
    reward_constant = -10
    end_states = [239]
    final_distribution = [[239], [1]]
    regularization_constant = 5
    my_grid_world = GridWorld(num_of_rows, num_of_cols, init_state_dist, obstacles, slip_probability, reward_constant)

    time_product_grid_world = TimeProductMDP(my_grid_world, 7, 'continue')

    #initial_state_dist = np.zeros(len(time_product_grid_world.list_of_states_and_transitions))
    #initial_state_dist[22] = 1
    val, _,  x_sa = TotalCostMDP.maximize_reward_with_concave_regularizer(time_product_grid_world, end_states, 'entropy',
                                                          regularization_constant, time_product_grid_world.initial_state_dist)


    # Save the constructed gridworld
    save_file_str = os.path.join(os.path.abspath(os.path.curdir), 'logs', 'time_product_mdp_results', '3stepcontinueMDP.pkl')
    with open(save_file_str, 'wb') as f:
        save_dict = {}
        save_dict['time_product_mdp'] = time_product_grid_world
        save_dict['x_sa'] = x_sa
        pickle.dump(save_dict, f)



if __name__ == '__main__':
    mymain()
