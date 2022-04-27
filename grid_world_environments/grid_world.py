import numpy as np

class GridWorld:
    def __init__(self,
                 num_of_rows: int,
                 num_of_cols: int,
                 obstacles: list,
                 slip_probability=0):

        self.rows = num_of_rows
        self.cols = num_of_cols
        self.obstacles = obstacles
        self.slip_probability = slip_probability
        self.NS = num_of_rows * num_of_cols


    def construct_transitions(self):
        list_of_states_and_transitions = []

        for state in range(self.NS):
            if state in self.obstacles:
                successors = np.array([[state]])
                transitions = np.eye(1)
                list_of_states_and_transitions.append([successors, transitions])
            else:
                location_of_the_state = self.verify_location(state)

                if location_of_the_state == "bottom left corner":
                    successors = np.array([[state+1, state+self.cols, state+self.cols+1]])
                    # Available actions: {up, right}
                    transitions = np.array([[0, 1-self.slip_probability, self.slip_probability],
                                            [1-self.slip_probability, 0, self.slip_probability]])
                    list_of_states_and_transitions.append([successors, transitions])

                elif location_of_the_state == "bottom right corner":
                    successors = np.array([[state-1, state+self.cols-1, state+self.cols]])
                    # Available actions: {up, left}
                    transitions = np.array([[0, self.slip_probability, 1-self.slip_probability],
                                            [1-self.slip_probability, self.slip_probability, 0]])
                    list_of_states_and_transitions.append([successors, transitions])

                elif location_of_the_state == "top left corner":
                    successors = np.array([state-self.cols, state-self.cols+1, state+1])
                    # Available actions = {bottom, right}
                    transitions = np.array([[1-self.slip_probability, self.slip_probability, 0],
                                            [0, self.slip_probability, 1-self.slip_probability]])
                    list_of_states_and_transitions.append([successors, transitions])

                elif location_of_the_state == "top right corner":
                    successors = np.array([state-self.cols-1, state-self.cols, state-1])
                    # Available actions = {bottom, left}
                    transitions = np.array([[self.slip_probability, 1-self.slip_probability, 0],
                                            [self.slip_probability, 0, 1-self.slip_probability]])
                    list_of_states_and_transitions.append([successors, transitions])

                elif location_of_the_state == "bottom":
                    successors = np.array([state-1, state+1, state+self.cols-1, state+self.cols, state+self.cols+1])
                    # Available actions = {up, left, right}
                    transitions = np.array([[0, 0, self.slip_probability/2, 1-self.slip_probability, self.slip_probability/2],
                                            [1-self.slip_probability, 0, self.slip_probability, 0, 0],
                                            [0, 1-self.slip_probability, 0, 0, self.slip_probability]])
                    list_of_states_and_transitions.append([successors, transitions])

                elif location_of_the_state == "top":
                    successors = np.array([state-self.cols-1, state-self.cols, state-self.cols+1, state-1, state+1])
                    # Available actions = {bottom, left, right}
                    transitions = np.array([[self.slip_probability/2, 1-self.slip_probability, self.slip_probability/2, 0, 0],
                                            [self.slip_probability, 0, 0, 1-self.slip_probability, 0],
                                            [0, 0, self.slip_probability, 0, 1-self.slip_probability]])
                    list_of_states_and_transitions.append([successors, transitions])

                elif location_of_the_state == "left":
                    successors = np.array([state-self.cols, state-self.cols+1, state+1, state+self.cols, state+self.cols+1])
                    # Available actions = {up, down, right}
                    transitions = np.array([[0, 0, 0, 1-self.slip_probability, self.slip_probability],
                                            [1-self.slip_probability, self.slip_probability, 0, 0, 0],
                                            [0, 1-self.slip_probability/2, 1-self.slip_probability, 0, self.slip_probability/2]])
                    list_of_states_and_transitions.append([successors, transitions])

                elif location_of_the_state == "right":
                    successors = np.array([state-self.cols-1, state-self.cols, state-1, state+self.cols-1, state+self.cols])
                    # Available actions = {up, down, left}
                    transitions = np.array([[0, 0, 0, self.slip_probability, 1-self.slip_probability],
                                            [self.slip_probability, 1-self.slip_probability, 0, 0, 0],
                                            [1-self.slip_probability/2, 0, 1-self.slip_probability, self.slip_probability/2, 0]])
                    list_of_states_and_transitions.append([successors, transitions])

                elif location_of_the_state == "middle":
                    successors = np.array([state-self.cols-1, state-self.cols, state-self.cols+1,
                                           state-1, state+1, state+self.cols-1, state+self.cols, state+self.cols+1])
                    # Available actions = {up, down, right, left}
                    transitions = np.array([[0, 0, 0, 0, 0, self.slip_probability/2, 1-self.slip_probability, self.slip_probability/2],
                                            [self.slip_probability/2, 1-self.slip_probability, self.slip_probability/2, 0, 0, 0, 0, 0],
                                            [self.slip_probability/2, 0, 0, 1-self.slip_probability, 0, self.slip_probability/2, 0, 0],
                                            [0, 0, self.slip_probability/2, 0, 1-self.slip_probability, 0, 0, self.slip_probability/2]])
                    list_of_states_and_transitions.append([successors, transitions])

        return list_of_states_and_transitions

    def verify_location(self, state):

        row_number = state // self.cols
        col_number = state % self.cols
        if row_number == 0 and col_number == 0:
            return "bottom left corner"
        if row_number == 0 and col_number == self.cols-1:
            return "bottom right corner"
        if row_number == self.rows-1 and col_number == 0:
            return "top left corner"
        if row_number == self.rows-1 and col_number == self.cols-1:
            return "top right corner"
        if row_number == 0:
            return "bottom"
        if row_number == self.rows-1:
            return "top"
        if col_number == 0:
            return "left"
        if col_number == self.cols-1:
            return "right"
        return "middle"
