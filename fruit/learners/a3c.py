import multiprocessing
from fruit.learners.base import Learner
from fruit.monitor.monitor import AgentMonitor
import numpy as np
from fruit.utils.annealer import Annealer


class A3CLearner(Learner):
    def __init__(self, agent, name, environment, network, global_dict, report_frequency,
                 network_update_steps=5, reward_clip_thresholds=(-1, 1)
                 ):
        super().__init__(agent=agent, name=name, environment=environment, network=network,
                         global_dict=global_dict,
                         report_frequency=report_frequency)
        self.async_update_steps = network_update_steps
        self.reward_clip_thresholds = reward_clip_thresholds
        self.initial_learning_rate = network.get_config().get_initial_learning_rate()
        self.current_learning_rate = self.initial_learning_rate
        self.learning_rate_annealer = Annealer(self.initial_learning_rate, 0, self.agent.max_training_steps)

    @staticmethod
    def get_default_number_of_learners():
        return multiprocessing.cpu_count()

    def reset(self):
        super().reset()
        self.reset_batch()

    def reset_batch(self):
        self.data_dict['states'] = []
        self.data_dict['actions'] = []
        self.data_dict['rewards'] = []
        self.data_dict['next_states'] = []
        self.data_dict['terminals'] = []

    def get_action(self, state):
        probs = self.get_probs(state)
        action_probs = probs - np.finfo(np.float32).epsneg
        try:
            sample = np.random.multinomial(1, action_probs)
            action_index = int(np.nonzero(sample)[0])
        except:
            print('Select greedy action', action_probs)
            action_index = np.argmax(probs)
        return action_index

    def update(self, state, action, reward, next_state, terminal):
        if self.history_length > 1:
            self.frame_buffer.add_state(state)

        if self.reward_clip_thresholds is not None:
            reward = np.clip(reward, self.reward_clip_thresholds[0], self.reward_clip_thresholds[1])

        if not self.testing:
            if self.history_length > 1:
                current_s = self.frame_buffer.get_buffer()[0]
                next_s = self.frame_buffer.get_buffer_add_state(next_state)[0]
            else:
                current_s = state
                next_s = next_state
            self.data_dict['states'].append(current_s)
            self.data_dict['actions'].append(action)
            self.data_dict['rewards'].append(reward)
            self.data_dict['next_states'].append(next_s)
            self.data_dict['terminals'].append(terminal)

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