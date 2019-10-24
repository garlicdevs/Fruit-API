from fruit.learners.a3c import A3CLearner
import numpy as np
from fruit.utils.annealer import Annealer
import tensorflow as tf


class DQA3CLearner(A3CLearner):
    def __init__(self, agent, name, environment, network, global_dict, report_frequency,
                 network_update_steps=5, reward_clip_thresholds=(-1, 1), auxiliary_model_path=None,
                 alpha=0.2, epsilon=0.02
                 ):
        super().__init__(agent=agent, name=name, environment=environment, network=network,
                         global_dict=global_dict,
                         report_frequency=report_frequency)
        self.async_update_steps = network_update_steps
        self.reward_clip_thresholds = reward_clip_thresholds
        self.initial_learning_rate = network.get_config().get_initial_learning_rate()
        self.current_learning_rate = self.initial_learning_rate
        self.learning_rate_annealer = Annealer(self.initial_learning_rate, 0, self.agent.max_training_steps)
        self.auxiliary_model_path = auxiliary_model_path
        self.alpha = alpha
        self.epsilon = epsilon
        self.load_model()

    def load_model(self):
        self.network.tf_saver.restore(self.network.tf_session, self.auxiliary_model_path)
        self.network.network_config.update_dq_network(self.network.tf_session)
        self.network.tf_saver = tf.train.Saver(var_list=self.network.tf_network_variables,
                                               max_to_keep=self.network.num_of_checkpoints)
        self.network.load_model()

    def get_action(self, state):
        probs = self.get_probs(state)
        probs_1 = probs[0]
        probs_2 = probs[1]
        gen = np.random.uniform(0, 1)
        if gen < self.epsilon:
            return np.random.randint(0, self.network.network_config.get_output_size())
        else:
            gen = np.random.uniform(0, 1)
            if gen < self.alpha:
                action_probs = probs_1 - np.finfo(np.float32).epsneg
                sample = np.random.multinomial(1, action_probs)
                action_index = int(np.nonzero(sample)[0])
                return action_index
            else:
                action_probs = probs_2 - np.finfo(np.float32).epsneg
                sample = np.random.multinomial(1, action_probs)
                action_index = int(np.nonzero(sample)[0])
                return action_index
