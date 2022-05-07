from markov_decision_processes.mdp import MDP
#from markov_decision_processes.averagemdp import AverageMDP
from markov_decision_processes.totalcostmdp import TotalCostMDP
from markov_decision_processes.timeproductmdp import TimeProductMDP
from grid_world.gridworld import GridWorld
import numpy as np

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


    num_of_rows = 5
    num_of_cols = 5
    num_of_states = num_of_rows*num_of_cols
    obstacles = []
    slip_probability = 0.001
    init_state_dist = np.asarray([0.0]*num_of_states); init_state_dist[3] = 0.5; init_state_dist[1] = 0.5
    reward_constant = -10
    end_states = [74]
    final_distribution = [[74], [1]]
    regularization_constant = 5
    my_grid_world = GridWorld(num_of_rows, num_of_cols, init_state_dist, obstacles, slip_probability, reward_constant)

    time_product_grid_world = TimeProductMDP(my_grid_world, 2, 'continue')

    initial_state_dist = np.zeros(len(time_product_grid_world.list_of_states_and_transitions))
    initial_state_dist[22] = 1
    TotalCostMDP.maximize_reward_with_concave_regularizer(time_product_grid_world, end_states, 'none',
                                                          regularization_constant, initial_state_dist)





if __name__ == '__main__':
    mymain()
