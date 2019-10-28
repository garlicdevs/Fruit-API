import threading
from sys import platform as _platform
from fruit.buffers.buffer import StateBuffer
from fruit.monitor.monitor import AgentMonitor
import numpy as np


class Learner(threading.Thread):
    """
    Users should define an RL algorithm that is a subclass of Learner.
    """
    def __init__(self, agent, name, environment, network, global_dict, report_frequency=1):
        super().__init__()

        range, is_range = environment.get_action_space().get_range()
        if not is_range:
            raise ValueError("Does not support this type of action space")

        self.step_count = 0
        self.eps_count = 0

        self.environment = environment
        self.name = name
        self.agent = agent

        self.num_actions = len(range)

        self.network = network
        self.config = network.network_config if network is not None else None
        self.history_length = self.config.get_history_length() if self.config is not None else 1

        if self.history_length > 1 and self.config is not None:
            self.frame_buffer = StateBuffer(self.config.get_input_shape(), history_length=self.history_length)

        self.global_dict = global_dict
        self.report_frequency = report_frequency

        self.testing = False

        self.initialize()

    def get_probs(self, state):
        if self.history_length > 1:
            probs = self.network.predict(self.frame_buffer.get_buffer_add_state(state))
        else:
            probs = self.network.predict(state)
        return probs

    def initialize(self):
        self.data_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'terminals': [],
            'learning_rate': self.network.get_config().get_initial_learning_rate() if self.network is not None else 'NA',
            'logging': False,
            'global_step': 0
        }

    def run_episode(self):
        """
        Run an episode

        :return: a total reward of the episode
        """
        self.environment.reset()
        self.reset()

        objs = self.environment.get_number_of_objectives()

        if objs <= 1:
            total_reward = 0
        else:
            total_reward = [0] * objs
        state = self.environment.get_state()
        terminal = False

        while not terminal:
            action = self.get_action(state)
            reward = self.environment.step(action)
            if objs <= 1:
                total_reward += reward
            else:
                total_reward = np.add(total_reward, reward)
            next_state = self.environment.get_state()
            terminal = self.environment.is_terminal()
            self.update(state, action, reward, next_state, terminal)
            state = next_state

        self.episode_end()
        return total_reward

    def episode_end(self):
        pass

    def reset(self):
        self.testing = self.agent.is_testing_mode

        if self.network is not None:
            self.network.reset_network()

        if self.history_length > 1:
            self.frame_buffer.reset()

            state = self.environment.get_state()
            for _ in range(self.history_length):
                self.frame_buffer.add_state(state)

    def report(self, reward):
        print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:',
              self.environment.get_current_steps(), 'Step count:', self.step_count, 'Learning rate:',
              self.global_dict[AgentMonitor.Q_LEARNING_RATE])

    def _run(self):
        reward = self.run_episode()
        self.eps_count += 1
        self.global_dict[AgentMonitor.Q_ADD_REWARD](reward, self.environment.get_current_steps())

        if self.eps_count % self.report_frequency == 0:
            self.report(reward)

    def run(self):
        if self.environment.is_render and _platform == "darwin":
            self._run()
        else:
            while not self.global_dict[AgentMonitor.Q_FINISH]:
                self._run()

    def update(self, state, action, reward, next_state, terminal):
        pass

    def get_action(self, state):
        pass
