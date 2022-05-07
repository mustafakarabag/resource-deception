import numpy as np
from markov_decision_processes.mdp import MDP

class TimeProductMDP(MDP):
    """
    A class to create product MDPs using a finite time horizon.
    """

    def __init__(self,
                 original_mdp: MDP,
                 T: int,
                 end_type: str):
        """
        Creates a product MDP using a given MDP and a time horizon.

        @param original_mdp: MDP that the product is build upon
        @param T: Time horizon. The agent has T active decisions. Must be greater than 0
        @param end_type: What happens when the time horizon ends.
                        'cut' cuts the time horizon and connects the states to themselves at the end.
                        'continue' connects the states to their original successor states.
        """

        if T < 1: raise ValueError('Time horizon must be positive.')

        self.original_mdp = original_mdp
        list_of_states_and_transitions = []
        self.T = T

        if original_mdp.reward is not None:
            self.reward = original_mdp.reward*(T+1)
        else:
            self.reward = None

        for t in range(T):
            for s in range(original_mdp.NS):
                succ_states = original_mdp.list_of_states_and_transitions[s][0] + original_mdp.NS*(t+1)
                list_of_states_and_transitions.append([succ_states, original_mdp.list_of_states_and_transitions[s][1]])


        if end_type == 'cut':
            for s in range(original_mdp.NS):
                #Connect the states themselves at the end
                succ_states = np.array([[original_mdp.NS*T + s]])
                list_of_states_and_transitions.append([succ_states, np.array([[1.0]])])
            self.end_type = end_type

        elif end_type == 'continue':
            for s in range(original_mdp.NS):
                #Connect states to their actual successors at the end
                succ_states = original_mdp.list_of_states_and_transitions[s][0] + original_mdp.NS*(T)
                list_of_states_and_transitions.append([succ_states, original_mdp.list_of_states_and_transitions[s][1]])
            self.end_type = end_type
        else:
            raise ValueError('End type is not recognized.')


        initial_state_dist = np.zeros(len(list_of_states_and_transitions))
        initial_state_dist[0:original_mdp.NS] = original_mdp.initial_state_dist


        # Initializes the resulting joint MDP if there are no pairs of MDPs to be merged
        super().__init__(list_of_states_and_transitions, initial_state_dist, self.reward)

    def product_state_to_original_state(self, product_index: int):
        """

        @param product_index:
        @return:
        """
        if (product_index > 0) and (product_index < self.NS):
            time_index = product_index//self.original_mdp.NS
            state_index = product_index % self.original_mdp.NS
            return (state_index, time_index)
        else:
            raise ValueError('Product state index is not valid.')
