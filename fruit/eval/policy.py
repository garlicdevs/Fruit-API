import numpy as np


class Policy(object):
    def select_action(self, current_state, action_space):
        pass

    def get_max_steps(self):
        pass


class RandomPolicy(Policy):
    def __init__(self, max_steps):
        self.max_steps = max_steps

    def select_action(self, current_state, action_space):
        range, is_range = action_space.get_range()
        if is_range:
            return np.random.randint(0, len(range))
        else:
            rand = np.random.uniform(0., 1.)
            return range[0]*rand + range[1]*(1-rand)

    def get_max_steps(self):
        return self.max_steps


class NoOpPolicy(Policy):
    def __init__(self, max_steps):
        self.max_steps = max_steps

    def select_action(self, current_state, action_space):
        range, is_range = action_space.get_range()
        if is_range:
            return 0
        else:
            rand = 0.5
            return range[0] * rand + range[1] * (1 - rand)

    def get_max_steps(self):
        return np.random.randint(1, self.max_steps)