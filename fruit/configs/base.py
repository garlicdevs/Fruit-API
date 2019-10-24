from fruit.networks.optimizer import OptimizerFactory
from fruit.networks.manager import LayerManager


class Config(object):
    def __init__(self, environment, initial_learning_rate=0.004, history_length=4,
                 debug_mode=False, gamma=0.99, optimizer=None):
        self.env = environment
        self.is_atari = environment.is_atari()
        self.action_space = environment.get_action_space()
        self.state_space = environment.get_state_space()
        self.action_range, self.is_action_discrete = self.action_space.get_range()
        self.state_range, self.is_state_discrete = self.state_space.get_range()
        self.initial_learning_rate = initial_learning_rate
        self.history_length = history_length
        if optimizer is not None:
            optimizer = OptimizerFactory.get_optimizer(method=optimizer)
        else:
            optimizer = OptimizerFactory.get_optimizer()
        self.optimizer = optimizer
        self.debug_mode = debug_mode
        self.gamma = gamma
        self.layer_manager = LayerManager()

    def get_output_size(self):
        if self.is_action_discrete:
            return len(self.action_range)
        else:
            return list(self.action_space.shape)[0]

    def get_input_shape(self):
        return list(self.state_space.shape)

    def get_initial_learning_rate(self):
        return self.initial_learning_rate

    def get_history_length(self):
        return self.history_length

    def get_optimizer(self):
        return self.optimizer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_debug_mode(self):
        return self.debug_mode

    # User defined
    def init_config(self):
        pass

    # User defined
    def reset_config(self):
        pass

    # User defined
    def predict(self, session, state):
        pass

    # User defined
    def train(self, session, data_dict):
        pass

    def get_params(self, data_dict):
        states = data_dict['states']
        actions = data_dict['actions']
        rewards = data_dict['rewards']
        next_states = data_dict['next_states']
        terminals = data_dict['terminals']
        learning_rate = data_dict['learning_rate']
        logging = data_dict['logging']
        global_step = data_dict['global_step']
        return states, actions, rewards, next_states, terminals, learning_rate, logging, global_step
