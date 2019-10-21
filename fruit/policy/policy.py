import numpy as np


class StartUpPolicy(object):
    def get_max_steps(self):
        pass

    def step(self, state, action_space):
        pass


class AtariStartupPolicy(StartUpPolicy):
    def __init__(self, max_steps=30):
        self.__max_steps = max_steps

    def get_max_steps(self):
        return np.random.randint(1, self.__max_steps)

    def step(self, state, action_space):
        max_action = action_space.get_max()
        rand_action = np.random.randint(0, max_action + 1)
        return rand_action
