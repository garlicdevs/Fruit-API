
class BaseEngine(object):
    """
    Any game engine should follow this interface.
    """
    def get_game_name(self):
        """
        Returns name of the game
        """
        pass

    def clone(self):
        """
        Clone itself
        """
        pass

    def get_num_of_objectives(self):
        """
        The number of objectives, i.e., the number of reward signals in the game
        """
        pass

    def get_num_of_agents(self):
        """
        The number of agents in the game
        """
        pass

    def reset(self):
        """
        Reset the episode
        """
        pass

    def step(self, action):
        """
        Ask agent to execute the specified ``action``
        """
        pass

    def render(self):
        """
        Draw GUI
        """
        pass

    def get_state(self):
        """
        Get current state (can be in graphical format)
        """
        pass

    def is_terminal(self):
        """
        Is the episode terminated?
        """
        pass

    def get_state_space(self):
        """
        Get the state space
        """
        pass

    def get_action_space(self):
        """
        Get the action space
        """
        pass

    def get_num_of_actions(self):
        """
        The number of possible actions that can be executed
        """
        pass
