import numpy as np


class Policy(object):
    """
    A policy is used right after starting an episode. This policy is used to evaluate the robustness of
    a deep RL method. This class is abstract and cannot be instantiated.

    :param max_steps: this policy only operates until ``max_steps`` reaches
    """
    def __init__(self, max_steps):
        self.max_steps = max_steps

    def select_action(self, current_state, action_space):
        """
        Select a next action in the ``action_space``, given the ``current_state``.

        :param current_state: the current state of the environment
        :param action_space: the action space of the environment
        :return: the next action in the ``action_space``
        """
        pass

    def get_max_steps(self):
        """
        Return the number of maximum steps from this policy.

        :return: the number of maximum steps
        """
        pass


class RandomPolicy(Policy):
    """
    ``RandomPolicy`` is a subclass of ``Policy``, which is used to select a random action.
    """
    def select_action(self, current_state, action_space):
        """
        See ``Policy::select_action()`` description
        """
        range, is_range = action_space.get_range()
        if is_range:
            return np.random.randint(0, len(range))
        else:
            rand = np.random.uniform(0., 1.)
            return range[0]*rand + range[1]*(1-rand)

    def get_max_steps(self):
        """
        See ``Policy::get_max_steps()`` description
        """
        return self.max_steps


class NoOpPolicy(Policy):
    def select_action(self, current_state, action_space):
        """
        See ``Policy::select_action()`` description
        """
        range, is_range = action_space.get_range()
        if is_range:
            return 0
        else:
            rand = 0.5
            return range[0] * rand + range[1] * (1 - rand)

    def get_max_steps(self):
        """
        See ``Policy::get_max_steps()`` description
        """
        return np.random.randint(1, self.max_steps)