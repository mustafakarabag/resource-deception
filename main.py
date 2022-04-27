from markov_decision_processes.mdp import MDP
#from markov_decision_processes.averagemdp import AverageMDP
#from markov_decision_processes.totalcostmdp import TotalCostMDP
from markov_decision_processes.timeproductmdp import TimeProductMDP
from grid_world_environments.grid_world import GridWorld
import numpy as np

def mymain():

    my_mdp = MDP([[np.array([[0,1,2,3]]),np.eye(4)],
                  [np.array([[0,1,2,3]]),np.eye(4)],
                    [np.array([[2]]), np.eye(1)],
                    [np.array([[3]]),np.eye(1)]], 0)

    reward_list = [
        np.array([-1.0, -1.0, 0.0, 0.0]),
        np.array([-1.0, -1.0, 0.0, 0.0]),
        np.array([0.0]),
        np.array([0.0])
    ]
    end_states = [2,3]
    final_state_const = ([2,3], np.array([0.2, 0.8]))
    initial_state_dist = np.array([0.2, 0.8, 0, 0])
    print(my_mdp)
    #TotalCostMDP.maximize_reward_with_concave_regularizer(my_mdp, reward_list, end_states, 'entropy', 0.1, initial_state_dist, final_state_const)

    my_mdp2 = MDP([[np.array([[0,1]]),np.eye(2)],
                  [np.array([[1,2,3]]),np.eye(3)],
                    [np.array([[0,1,2,3]]),np.eye(4)],
                    [np.array([[0,1,2]]),np.eye(3)]], 0)

    time_product_MDP = TimeProductMDP(my_mdp2, 2, 'continue')
    print(time_product_MDP.product_state_to_original_state(8))
    print(time_product_MDP.NS)


    num_of_rows = 5
    num_of_cols = 5
    obstacles = []
    slip_probability = 0.1
    init_state = 0
    my_grid_world = GridWorld(num_of_rows, num_of_cols, obstacles, slip_probability)
    grid_transitions = my_grid_world.construct_transitions()
    my_mdp3 = MDP(grid_transitions, init_state)
    time_product_MDP = TimeProductMDP(my_mdp3, 2, 'continue')
    print(time_product_MDP.product_state_to_original_state(38))
    print(time_product_MDP.NS)



if __name__ == '__main__':
    mymain()
