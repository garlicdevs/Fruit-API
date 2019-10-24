from fruit.learners.a3c import A3CLearner
from fruit.learners.dqn import DQNLearner
import numpy as np

from fruit.monitor.monitor import AgentMonitor


class MODQNLearner(DQNLearner):
    def __init__(self, agent, name, environment, network, global_dict, report_frequency,
                 batch_size=32, warmup_steps=10000, training_frequency=1, experience_replay_size=100000,
                 epsilon_annealing_start=1, epsilon_annealing_end=0, reward_clip_thresholds=None
                 ):

        super().__init__(agent=agent, name=name, environment=environment, network=network, global_dict=global_dict,
                         report_frequency=report_frequency, batch_size=batch_size, warmup_steps=warmup_steps,
                         training_frequency=training_frequency, experience_replay_size=experience_replay_size,
                         epsilon_annealing_start=epsilon_annealing_start, epsilon_annealing_end=epsilon_annealing_end,
                         epsilon_annealing_steps=agent.max_training_steps,
                         reward_clip_thresholds=reward_clip_thresholds)


class MOSCA3CLearner(A3CLearner):
    def __init__(self, agent, name, environment, network, global_dict, report_frequency,
                 network_update_steps=5, reward_clip_thresholds=(-1, 1)):
        self.total_weights = len(network.network_config.weights)
        self.thread_id = int(name.split(' ')[-1]) % self.total_weights
        super().__init__(agent=agent, name=name, environment=environment, network=network, global_dict=global_dict,
                         report_frequency=report_frequency, network_update_steps=network_update_steps,
                         reward_clip_thresholds=reward_clip_thresholds)

    def initialize(self):
        self.data_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'terminals': [],
            'learning_rate': self.network.get_config().get_initial_learning_rate() if self.network is not None else 'NA',
            'logging': False,
            'global_step': 0,
            'thread_id': self.thread_id
        }

    def get_probs(self, state):
        if self.history_length > 1:
            probs = self.network.predict([self.frame_buffer.get_buffer_add_state(state), self.thread_id])
        else:
            probs = self.network.predict([state, self.thread_id])
        return probs