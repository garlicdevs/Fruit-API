from fruit.buffers.buffer import StateBuffer
from fruit.learners.a3c import A3CLearner
from fruit.monitor.monitor import AgentMonitor
import numpy as np


class A3CMapLearner(A3CLearner):
    def __init__(self, agent, name, environment, network, global_dict, report_frequency,
                 network_update_steps=5, reward_clip_thresholds=(-1, 1), update_reward_fnc=None):
        super().__init__(agent=agent, name=name, environment=environment, network=network, global_dict=global_dict,
                         report_frequency=report_frequency, network_update_steps=network_update_steps,
                         reward_clip_thresholds=reward_clip_thresholds)
        if self.history_length > 1 and self.config is not None:
            self.map_buffer = StateBuffer(self.config.get_input_shape(), history_length=self.history_length)
        self.update_reward_fnc = update_reward_fnc

    def reset_batch(self):
        self.data_dict['states'] = []
        self.data_dict['actions'] = []
        self.data_dict['rewards'] = []
        self.data_dict['next_states'] = []
        self.data_dict['terminals'] = []
        self.data_dict['map_data'] = []

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
            'map_data': []
        }

    def reset(self):
        self.testing = self.agent.is_testing_mode

        if self.network is not None:
            self.network.reset_network()

        if self.history_length > 1:
            self.frame_buffer.reset()
            self.map_buffer.reset()

            state = self.environment.get_state()
            map_data = self.environment.get_processor().get_map()

            for _ in range(self.history_length):
                self.frame_buffer.add_state(state)
                self.map_buffer.add_state(map_data)

        self.reset_batch()

    def get_probs_map(self, state, map_data):
        if self.history_length > 1:
            probs = self.network.predict([self.frame_buffer.get_buffer_add_state(state),
                                          self.map_buffer.get_buffer_add_state(map_data)])
        else:
            probs = self.network.predict([state, map_data])
        return probs

    def run_episode(self):
        self.environment.reset()
        self.reset()

        objs = self.environment.get_number_of_objectives()
        if objs <= 1:
            total_reward = 0
        else:
            total_reward = [0] * objs

        state = self.environment.get_state()
        map_data = self.environment.get_processor().get_map()
        terminal = False

        while not terminal:
            action = self.get_action_map(state, map_data)
            reward = self.environment.step(action)

            if objs <= 1:
                total_reward += reward
            else:
                total_reward = np.add(total_reward, reward)

            update_reward = total_reward
            if self.update_reward_fnc is not None:
                update_reward = self.update_reward_fnc(reward)

            next_state = self.environment.get_state()
            next_map = self.environment.get_processor().get_map()

            terminal = self.environment.is_terminal()

            self.update_map(state, action, update_reward, next_state, terminal, map_data)
            state = next_state
            map_data = next_map

        self.episode_end()
        return total_reward

    def get_action_map(self, state, map_data):
        probs = self.get_probs_map(state, map_data)
        action_probs = probs - np.finfo(np.float32).epsneg
        try:
            sample = np.random.multinomial(1, action_probs)
            action_index = int(np.nonzero(sample)[0])
        except:
            print('Select greedy action', action_probs)
            action_index = np.argmax(probs)
        return action_index

    def update_map(self, state, action, reward, next_state, terminal, map_data):
        if self.history_length > 1:
            self.frame_buffer.add_state(state)
            self.map_buffer.add_state(map_data)

        if self.reward_clip_thresholds is not None:
            reward = np.clip(reward, self.reward_clip_thresholds[0], self.reward_clip_thresholds[1])

        if not self.testing:
            if self.history_length > 1:
                current_s = self.frame_buffer.get_buffer()[0]
                next_s = self.frame_buffer.get_buffer_add_state(next_state)[0]
                map_d = self.map_buffer.get_buffer()[0]
            else:
                current_s = state
                next_s = next_state
                map_d = map_data
            self.data_dict['states'].append(current_s)
            self.data_dict['actions'].append(action)
            self.data_dict['rewards'].append(reward)
            self.data_dict['next_states'].append(next_s)
            self.data_dict['terminals'].append(terminal)
            self.data_dict['map_data'].append(map_d)

        self.step_count += 1
        self.global_dict[AgentMonitor.Q_GLOBAL_STEPS] += 1

        if not self.testing:
            if self.step_count % self.async_update_steps == 0 or terminal:
                logging = self.global_dict[AgentMonitor.Q_LOGGING]
                self.current_learning_rate = self.learning_rate_annealer.anneal(
                    self.global_dict[AgentMonitor.Q_GLOBAL_STEPS])
                self.data_dict['learning_rate'] = self.current_learning_rate
                self.global_dict[AgentMonitor.Q_LEARNING_RATE] = self.current_learning_rate
                if logging:
                    self.global_dict[AgentMonitor.Q_LOGGING] = False
                    self.data_dict['logging'] = True
                    summary = self.network.train_network(self.data_dict)
                    self.global_dict[AgentMonitor.Q_WRITER].\
                        add_summary(summary, global_step=self.global_dict[AgentMonitor.Q_GLOBAL_STEPS])
                else:
                    self.data_dict['logging'] = False
                    self.network.train_network(self.data_dict)
                self.reset_batch()