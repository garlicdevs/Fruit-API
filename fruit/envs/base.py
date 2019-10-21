

# Derived class must follow this interface
class BaseEnvironment(object):

    def clone(self):
        pass

    def step(self, actions):
        pass

    def reset(self):
        pass

    def get_current_steps(self):
        pass

    def get_action_space(self):
        pass

    def get_state_space(self):
        pass

    def step_all(self, action):
        pass

    def get_state(self):
        pass

    def is_terminal(self):
        pass

    def is_atari(self):
        return False

    def is_render(self):
        pass

    def get_number_of_objectives(self):
        pass

    def get_number_of_agents(self):
        pass