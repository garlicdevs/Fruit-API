from fruit.buffers.replay import SyncExperienceReplay
from fruit.learners.base import Learner
import numpy as np

from fruit.monitor.monitor import AgentMonitor
from fruit.utils.annealer import Annealer

experience_replay = None


class DQNLearner(Learner):
    def __init__(self, agent, name, environment, network, global_dict, report_frequency,
                 batch_size=32, warmup_steps=50000, training_frequency=4, experience_replay_size=2**19,
                 epsilon_annealing_start=1, epsilon_annealing_end=0.1,
                 epsilon_annealing_steps=1e6, reward_clip_thresholds=(-1, 1)
                 ):
        super().__init__(agent=agent, name=name, environment=environment, network=network, global_dict=global_dict,
                         report_frequency=report_frequency)
        global experience_replay
        with global_dict[AgentMonitor.Q_LOCK]:
            if experience_replay is None:
                experience_replay = SyncExperienceReplay(experience_replay_size,
                                                         state_history=network.network_config.get_history_length())
        self.replay = experience_replay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.training_frequency = training_frequency
        self.reward_clip_thresholds = reward_clip_thresholds
        self.epsilon_annealer = Annealer(epsilon_annealing_start, epsilon_annealing_end, epsilon_annealing_steps)
        self.current_learning_rate = network.get_config().get_initial_learning_rate()
        self.current_epsilon = epsilon_annealing_start

    @staticmethod
    def get_default_number_of_learners():
        return 1

    def get_action(self, state):
        probs = self.get_probs(state)
        if self.current_epsilon is not None:
            if np.random.uniform(0, 1) < self.current_epsilon:
                return np.random.randint(0, len(probs))
            else:
                return np.argmax(probs)
        else:
            return np.argmax(probs)

    def report(self, reward):
        print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:',
              self.environment.get_current_steps(), 'Step count:', self.step_count, 'Learning rate:',
              self.global_dict[AgentMonitor.Q_LEARNING_RATE], 'Epsilon:', self.current_epsilon)

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
            self.replay.append(current_s, action, reward, next_s, terminal)

        self.step_count += 1
        self.global_dict['counter'] += 1

        if self.step_count < self.warmup_steps:
            return

        if not self.testing:
            if self.step_count % self.training_frequency == 0:
                logging = self.global_dict[AgentMonitor.Q_LOGGING]
                s, a, r, n, t = self.replay.get_mini_batch(batch_size=self.batch_size)
                self.data_dict['states'] = s
                self.data_dict['actions'] = a
                self.data_dict['rewards'] = r
                self.data_dict['next_states'] = n
                self.data_dict['terminals'] = t
                self.data_dict['learning_rate'] = self.current_learning_rate
                self.data_dict['global_step'] = self.global_dict[AgentMonitor.Q_GLOBAL_STEPS]
                if logging:
                    self.global_dict[AgentMonitor.Q_LOGGING] = False
                    self.data_dict['logging'] = True
                    summary = self.network.train_network(self.data_dict)
                    self.global_dict[AgentMonitor.Q_WRITER]. \
                        add_summary(summary, global_step=self.global_dict[AgentMonitor.Q_GLOBAL_STEPS])
                else:
                    self.data_dict['logging'] = False
                    self.network.train_network(self.data_dict)

            self.current_epsilon = self.epsilon_annealer.anneal(self.global_dict[AgentMonitor.Q_GLOBAL_STEPS])
