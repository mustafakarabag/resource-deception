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

        if T < 1: raise ValueError('Time horizon must be positive.')

        self.original_mdp = original_mdp
        list_of_states_and_transitions = []
        initial_state_index = original_mdp.initial_state_index
        self.T = T
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

        # Initializes the resulting joint MDP if there are no pairs of MDPs to be merged
        super().__init__(list_of_states_and_transitions, initial_state_index)

    def product_state_to_original_state(self, product_index: int):
        if (product_index > 0) and (product_index < self.NS):
            time_index = product_index//self.original_mdp.NS
            state_index = product_index % self.original_mdp.NS
            return (state_index, time_index)
        else:
            raise ValueError('Product state index is not valid.')
