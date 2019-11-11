

# Derived class must follow this interface
class BaseEnvironment(object):
    """
    ``BaseEnvironment`` defines a unique interface used by Fruit API.
    Therefore, to integrate external environments into the framework, it is
    necessary to create a subclass of ``BaseEnvironment`` and implement all functions
    declared in this class.
    """
    def clone(self):
        """
        Duplicate itself. The function is useful in RL methods where
        multiple learners are trained in different environments.
        """
        pass

    def step(self, actions):
        """
        Execute the next ``actions``.

        :param actions: next actions that will be executed.
        :return: return a set of rewards
        """
        pass

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        pass

    def get_current_steps(self):
        """
        Get the current number of steps.

        :return: the current number of steps
        """
        pass

    def get_action_space(self):
        """
        Get the action space of the environment

        :return: the action space
        """
        pass

    def get_state_space(self):
        """
        Get the state space of the environment

        :return: the state space
        """
        pass

    def step_all(self, action):
        """
        Similar to ``step()`` but returns verbose information.

        :param action: next actions that will be executed
        :return: next state, rewards, is terminal, debug info
        """
        pass

    def get_state(self):
        """
        Get current state of the environment.

        :return: the current state
        """
        pass

    def is_terminal(self):
        """
        Check if the episode is terminated.

        :return: True if the current episode is terminated else False
        """
        pass

    def is_atari(self):
        """
        Check if the environment is an Atari game

        :return: True if Atari game else False
        """
        return False

    def is_render(self):
        """
        Check if the environment shows GUI.

        :return: True if showing GUI else False
        """
        pass

    def get_number_of_objectives(self):
        """
        Get the number of objectives.

        :return: the number of objectives
        """
        pass

    def get_number_of_agents(self):
        """
        Get the number of agents in the environment.

        :return: the number of agents
        """
        pass

    def get_processor(self):
        """
        Get state processor

        :return: state processor
        """
        return None