from fruit.networks.optimizer import OptimizerFactory
from fruit.networks.manager import LayerManager


class Config(object):
    """
    A network's configuration, which defines network architecture, training step, and optimizer.
    A user-defined configuration should be a subclass of ``Config``.

    :param environment: the environment
    :param initial_learning_rate: the learning rate
    :param history_length: the number of historical states as a single state
    :param debug_mode: enable this flag to print verbose information
    :param gamma: the discounted factor
    :param optimizer: an optimizer (can be retrieved from OptimizerFactory)
    """
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
        """
        Get size of the network's output

        :return: output size
        """
        if self.is_action_discrete:
            return len(self.action_range)
        else:
            return list(self.action_space.shape)[0]

    def get_input_shape(self):
        """
        Get shape of the input

        :return: input shape
        """
        return list(self.state_space.shape)

    def get_initial_learning_rate(self):
        """
        Get initial learning rate

        :return: initial learning rate
        """
        return self.initial_learning_rate

    def get_history_length(self):
        """
        Get history length of a single state

        :return: history length
        """
        return self.history_length

    def get_optimizer(self):
        """
        Get the current optimizer used by the configuration

        :return: the current optimizer
        """
        return self.optimizer

    def set_optimizer(self, optimizer):
        """
        Set new optimizer for the current configuration

        :param optimizer: new optimizer
        """
        self.optimizer = optimizer

    def get_debug_mode(self):
        """
        Get debug mode flag

        :return: True if in debug mode else False
        """
        return self.debug_mode

    # User defined
    def init_config(self):
        """
        Create the network

        :return: parameters of the network
        """
        pass

    # User defined
    def reset_config(self):
        """
        Reset the configuration
        """
        pass

    # User defined
    def predict(self, session, state):
        """
        Evaluate the network by using a specified state.

        :param session: the current session id (from Tensorflow)
        :param state: a state
        """
        pass

    # User defined
    def train(self, session, data_dict):
        """
        Train the network

        :param session: the current session id (from Tensorflow)
        :param data_dict: a user-defined data dictionary sent by the learner
        """
        pass

    def get_params(self, data_dict):
        """
        Parse a user-defined data dictionary.

        :param data_dict: a user-define data dictionary
        :return: verbose information of ``data_dict``
        """
        states = data_dict['states']
        actions = data_dict['actions']
        rewards = data_dict['rewards']
        next_states = data_dict['next_states']
        terminals = data_dict['terminals']
        learning_rate = data_dict['learning_rate']
        logging = data_dict['logging']
        global_step = data_dict['global_step']
        return states, actions, rewards, next_states, terminals, learning_rate, logging, global_step
