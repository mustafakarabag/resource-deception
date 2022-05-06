import numpy as np
from markov_decision_processes.mdp import MDP

class GridWorld(MDP):
    def __init__(self,
                 num_of_rows: int,
                 num_of_cols: int,
                 obstacle_locations: list,
                 slip_probability=0):
        """

        @param num_of_rows: Number of rows of the grid
        @param num_of_cols: Number of columns of the grid
        @param obstacle_locations: List of 2-element arrays for the obstacle locations
        @param slip_probability: With this probability the agent slips to the locations next to the target state
        """
        self.Nrows = num_of_rows
        self.Ncols = num_of_cols
        self.obstacle_locations = obstacle_locations
        self.slip_probability = slip_probability
        self.NS = self.Nrows*self.Ncols
        self.list_of_states_and_transitions = []
        self.available_actions = []

        #Create the list of successor states and transitions
        for state in range(self.NS):
            loc = np.asarray(np.unravel_index(state, (self.Nrows, self.Ncols)))
            successor_locs = []

            #Find potential successor states
            for row_diff in range(-1,2):
                for col_diff in range(-1,2):
                    next_loc = loc + np.array([row_diff, col_diff])
                    if not (self.is_out_of_bounds(next_loc) or self.is_obstacle(next_loc) or all(loc == next_loc)):
                        successor_locs.append(next_loc)

            #Construct transitions
            act_list = ['U', 'D', 'L', 'R']
            tran_probs = np.zeros((len(act_list), len(successor_locs)))
            act_validity = [True] * len(act_list)
            valid_acts = []
            for act_ind in range(len(act_list)):
                next_loc = self.find_next_loc(loc, act_list[act_ind])
                if not (self.is_out_of_bounds(next_loc) or self.is_obstacle(next_loc) or all(loc == next_loc)):

                    #Set main target state probability
                    tran_probs[act_ind, self.find_unique_array_in_a_list(next_loc, successor_locs)] = 1 -self.slip_probability

                    #Find potential slip locations
                    slip_locs = []
                    for neigh_loc in successor_locs:
                        if np.sum(np.abs(neigh_loc - next_loc)) == 1:
                            slip_locs.append(neigh_loc)

                    #Set slip probabilities
                    for slip_loc in slip_locs:
                        tran_probs[act_ind, self.find_unique_array_in_a_list(slip_loc, successor_locs)] = self.slip_probability/len(slip_locs)

                    #If there is no slip state, add slip to the main target state
                    if len(slip_locs) == 0:
                        tran_probs[act_ind, successor_locs.index(next_loc)] = 1
                    valid_acts.append(act_list[act_ind])
                else:
                    act_validity[act_ind] = False


            if any(act_validity):
                tran_probs = tran_probs[act_validity]
                self.available_actions.append(valid_acts)
            else:
            #Make the state a loop state if there is no available action
                successor_locs = [loc]
                tran_probs = np.eye(1)
                self.available_actions.append(['Loop'])

            if self.is_obstacle(loc):
                successor_locs = [loc]
                tran_probs = np.eye(1)
                self.available_actions[-1] = ['Loop']

            #Find the state indices of the successor state locations
            succ_states = np.ravel_multi_index(np.transpose(np.array(successor_locs)), (self.Nrows, self.Ncols))
            self.list_of_states_and_transitions.append([np.array([succ_states]), tran_probs])




    def is_out_of_bounds(self, loc:np.ndarray) -> bool:
        """
        Checks whether the location is out of bounds
        @param loc: 1-dimensional, 2-element location array
        @return: boolean
        """
        if (0 <= loc[1] < self.Ncols) and (0 <= loc[0] < self.Nrows):
            return False
        else:
            return True

    def is_obstacle(self, loc:np.ndarray) -> bool:
        """
        Checks whether the location is an obstacle
        @param loc: 1-dimensional, 2-element location array
        @return: boolean
        """
        if self.find_unique_array_in_a_list(loc,self.obstacle_locations) != -1:
            return True
        else:
            return False

    def find_next_loc(self, loc, act:str):
        """
        Finds the next location. The next location might be out of bounds or an obstacle
        @param loc: 1-dimensional, 2-element location array
        @param act: A letter 'U', 'D', 'L', or 'R'
        @return: 1-dimensional, 2-element location array of the next location
        """
        if act == 'U':
            target_loc = loc + np.array([1, 0])
        elif act == 'D':
            target_loc = loc + np.array([-1, 0])
        elif act == 'L':
            target_loc = loc + np.array([0, 1])
        elif act == 'R':
            target_loc = loc + np.array([0, -1])
        else:
            raise ValueError('Not a valid action.')

        return target_loc

    def find_unique_array_in_a_list(self,arr:np.ndarray, list_of_arrays:list):
        """
        Returns the index of the array in the list. If the array is not in the list, returns -1
        @param arr: a numpy array
        @param list_of_arrays: a list of numpy arrays
        @return: index of the given array
        """
        for ind in range(len(list_of_arrays)):
            if (arr == list_of_arrays[ind]).all():
                return ind

        return -1
