import numpy as np


# A port from https://github.com/FedUni/MORL/blob/master/morlglue-clients/tools/valuefunction
class LookupTable(object):
    def __init__(self, environment, init_value=0.):
        self.num_of_objs = environment.get_number_of_objectives()
        r, _ = environment.get_action_space().get_range()
        self.num_of_actions = len(r)
        r, _ = environment.get_state_space().get_range()
        self.num_of_states = len(r)

        self.value_function = []
        for i in range(self.num_of_objs):
            element = [[0. for x in range(self.num_of_states)] for y in range(self.num_of_actions)]
            self.value_function.append(element)

        if init_value != 0:
            for i in range(self.num_of_objs):
                for j in range(self.num_of_actions):
                    for k in range(self.num_of_states):
                        self.value_function[i][j][k] = init_value

        self.errors = [0.] * self.num_of_objs

    def calculate_td_errors(self, action, prev_state, greedy_action, new_state, gamma, reward):
        for i in range(self.num_of_objs):
            values = self.value_function[i]
            current_q_values = values[action][prev_state]
            max_q_values = values[greedy_action][new_state]

            err = reward[i] + gamma * max_q_values - current_q_values

            self.errors[i] = err

    def calculate_terminal_td_errors(self, action, prev_state, gamma, reward):
        for i in range(self.num_of_objs):
            values = self.value_function[i]
            current_q_values = values[action][prev_state]
            self.errors[i] = reward[i] - current_q_values

    def update(self, action, state, lamb, alpha):
        for i in range(self.num_of_objs):
            values = self.value_function[i]
            current_q_values = values[action][state]
            new_q_values = current_q_values + alpha * (lamb * self.errors[i])
            self.value_function[i][action][state] = new_q_values

    def reset_q_values(self, init_value):
        for i in range(self.num_of_objs):
            for j in range(self.num_of_actions):
                for k in range(self.num_of_states):
                    self.value_function[i][j][k] = init_value[i]

    def get_q_values(self, action, state):
        result = [0.]*self.num_of_objs
        for i in range(self.num_of_objs):
            result[i] = self.value_function[i][action][state]
        return result

    def print_values(self):
        for i in range(self.num_of_objs):
            print("+ Objective: ", i)
            for j in range(self.num_of_states):
                s = "State " + str(j) + " :\t"
                for k in range(self.num_of_actions):
                    s = s + "{0:.1f}".format(self.value_function[i][k][j]) + "\t"
                print(s)

    def save_value_function(self, file_name):
        array = np.array(self.value_function)
        np.save(file_name, array)

    def load_value_function(self, file_name):
        array = np.load(file_name)
        self.num_of_objs = len(array)
        self.errors = [0.] * self.num_of_objs
        self.num_of_actions = len(array[0])
        self.num_of_states = len(array[0][0])
        self.value_function = []
        for i in range(self.num_of_objs):
            element = [[0. for x in range(self.num_of_states)] for y in range(self.num_of_actions)]
            self.value_function.append(element)
        for i in range(self.num_of_objs):
            for j in range(self.num_of_actions):
                for k in range(self.num_of_states):
                    self.value_function[i][j][k] = array[i][j][k]


class TLO(object):
    @staticmethod
    def compare(a, b, threshold):
        size = len(threshold)
        for i in range(size):
            th_a = min(a[i], threshold[i])
            th_b = min(b[i], threshold[i])
            if th_a > th_b:
                return 1
            elif th_a < th_b:
                return -1
        if a[size] > b[size]:
            return 1
        elif a[size] < b[size]:
            return -1
        for i in range(size):
            if a[i] > b[i]:
                return 1
            elif a[i] < b[i]:
                return -1
        return 0

    @staticmethod
    def greedy_action(action_values, thresholds):
        best_actions = [0]
        for i in range(len(action_values)):
            if i > 0:
                ret = TLO.compare(action_values[i], action_values[best_actions[0]], threshold=thresholds)
                if ret > 0:
                    best_actions.clear()
                    best_actions.append(i)
                elif ret == 0:
                    best_actions.append(i)
        if len(best_actions) > 1:
            return best_actions[np.random.randint(0, len(best_actions)-1)]
        else:
            return best_actions[0]


class TLOLookupTable(LookupTable):
    def __init__(self, environment, init_value=0., thresholds=None):
        super().__init__(environment=environment, init_value=init_value)
        self.thresholds = thresholds
        self.current_state_values = [[0. for x in range(self.num_of_objs)] for y in range(self.num_of_actions)]

    def get_thresholds(self):
        return self.thresholds

    def set_threshold(self, thresholds):
        self.thresholds = thresholds

    def select_greedy_action(self, state):
        self.get_action_values(state)
        return TLO.greedy_action(self.current_state_values, self.thresholds)

    def is_greedy(self, state, action):
        self.get_action_values(state)
        best = TLO.greedy_action(self.current_state_values, self.thresholds)
        return TLO.compare(self.current_state_values[action], self.current_state_values[best], self.thresholds) == 0

    def get_action_values(self, state):
        for i in range(self.num_of_objs):
            for a in range(self.num_of_actions):
                self.current_state_values[a][i] = self.value_function[i][a][state]


class LinearLookupTable(LookupTable):
    def __init__(self, environment, init_value=0., thresholds=None):
        super().__init__(environment=environment, init_value=init_value)
        self.thresholds = thresholds
        self.current_state_values = [[0. for x in range(self.num_of_objs)] for y in range(self.num_of_actions)]

    def get_thresholds(self):
        return self.thresholds

    def set_threshold(self, thresholds):
        self.thresholds = thresholds

    def select_greedy_action(self, state):
        self.get_action_values(state)
        return self.greedy_action(self.current_state_values, self.thresholds)

    def get_action_values(self, state):
        for i in range(self.num_of_objs):
            for a in range(self.num_of_actions):
                self.current_state_values[a][i] = self.value_function[i][a][state]

    @staticmethod
    def greedy_action(action_values, thresholds):
        linear_values = []
        for i in range(len(action_values)):
            linear_values.append(np.sum(np.multiply(action_values[i], thresholds)))
        greedy_action = 0
        greedy_value = linear_values[0]
        for i in range(len(linear_values)):
            if i > 0:
                if linear_values[i] > greedy_value:
                    greedy_action = i
        return greedy_action