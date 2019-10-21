import numpy as np


class TransitionPair(object):
    def __init__(self, state, prob):
        self.state = state
        self.prob = prob


class TransitionList(object):
    def __init__(self):
        self.list = []

    def add(self, state, prob):
        for pair in self.list:
            if pair.state == state:
                pair.prob = pair.prob + prob
                return
        self.list.append(TransitionPair(state, prob))

    def get_prob(self, state):
        for pair in self.list:
            if pair.state == state:
                return pair.prob
        return 0.

    def get_next_state(self):
        r = np.random.uniform(0., 1.)
        sum = 0.
        for i in range(len(self.list)-1):
            sum = sum + self.list[i].prob
            if sum > r:
                return self.list[i].state

        return self.list[-1].state