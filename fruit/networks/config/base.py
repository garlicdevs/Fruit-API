from functools import partial
import tensorflow as tf
from fruit.envs.gym import GymEnvironment
from fruit.networks.config.optimizer import OptimizerFactory
from fruit.state.processor import AtariProcessor
from fruit.networks.manager import LayerManager


class MapConfig(object):
    def __init__(self, environment, initial_learning_rate=0.004, history_length=4, debug_mode=False, gamma=0.99,
                 stochastic_policy=True, optimizer=partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)):
        self.env = environment
        self.is_atari = environment.is_atari()
        self.action_space = environment.get_action_space()
        self.state_space = environment.get_state_space()
        self.action_range, self.is_action_discrete = self.action_space.get_range()
        self.state_range, self.is_state_discrete = self.state_space.get_range()
        self.initial_learning_rate = initial_learning_rate
        self.history_length = history_length
        self.optimizer = optimizer
        self.debug_mode = debug_mode
        self.gamma = gamma
        self.stochastic = stochastic_policy
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

    def get_debug_mode(self):
        return self.debug_mode

    def is_stochastic(self):
        return self.stochastic

    # User defined
    def init_config(self):
        pass

    # User defined
    def reset_config(self):
        pass

    # User defined
    def get_prediction(self, session, state, map):
        pass

    # User defined
    def train_network(self, session, states, actions, rewards, next_states, terminals, maps,
                      learning_rate, summaries, global_step, other_data):
        pass


class MapA3CConfig(MapConfig):
    def __init__(self, environment, initial_learning_rate=0.004, history_length=4, beta=0.01,
                 global_norm_clipping=40, debug_mode=False, gamma = 0.99, stochastic=True,
                 optimizer=partial(tf.train.RMSPropOptimizer, decay=0.99, epsilon=0.1)):
        self.beta = beta
        self.global_norm_clipping = global_norm_clipping

        super().__init__(environment=environment, initial_learning_rate=initial_learning_rate,
                         history_length=history_length, debug_mode=debug_mode, gamma=gamma, stochastic_policy=stochastic,
                         optimizer=optimizer)


def __unit_test():
    environment = GymEnvironment("Breakout-v0", state_processor=AtariProcessor())
    config = Config(environment)
    print(config.get_input_shape())
    print(config.get_output_size())

    environment = GymEnvironment("CartPole-v0")
    config = Config(environment)
    print(config.get_input_shape())
    print(config.get_output_size())

    environment = GymEnvironment("Pendulum-v0")
    config = Config(environment)
    print(config.get_input_shape())
    print(config.get_output_size())

    environment = GymEnvironment("MountainCar-v0")
    config = Config(environment)
    print(config.get_input_shape())
    print(config.get_output_size())


if __name__ == '__main__':
    __unit_test()