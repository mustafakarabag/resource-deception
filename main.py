from markov_decision_processes.mdp import MDP
from markov_decision_processes.averagemdp import AverageMDP
from markov_decision_processes.totalcostmdp import TotalCostMDP
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
    TotalCostMDP.maximize_reward_with_concave_regularizer(my_mdp, reward_list, end_states, 'entropy', 0.1, initial_state_dist, final_state_const)

if __name__ == '__main__':
    mymain()


