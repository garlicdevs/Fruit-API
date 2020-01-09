from fruit.learners.base import Learner
from fruit.monitor.monitor import AgentMonitor
from fruit.utils.annealer import Annealer
import numpy as np


class TDLearner(Learner):
    def __init__(self, agent, name, environment, network, global_dict, report_frequency,
                 step_size=0.9, epsilon_annealing_start=0.9, epsilon_annealing_end=0,
                 load_model_path=None, using_e_greedy=True):
        super().__init__(agent=agent, name=name, environment=environment, network=network, global_dict=global_dict,
                         report_frequency=report_frequency)
        self.step_size = step_size
        self.epsilon_start = epsilon_annealing_start
        self.epsilon_end = epsilon_annealing_end
        self.load_model_path = load_model_path
        self.using_e_greedy = using_e_greedy
        r, _ = environment.get_state_space().get_range()
        self.num_of_states = len(r)
        self.table = [0.5 for _ in range(self.num_of_states)]
        self.epsilon_annealer = Annealer(epsilon_annealing_start, epsilon_annealing_end, self.agent.max_training_steps)
        self.current_epsilon = epsilon_annealing_start

    def report(self, reward):
        print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:',
              self.environment.get_current_steps(), 'Step count:', self.step_count, 'Learning rate:',
              self.step_size, 'Epsilon:', self.current_epsilon)

    def get_action(self, state):
        if self.using_e_greedy:
            if np.random.uniform(0, 1) <= self.current_epsilon:
                e_greedy = np.random.randint(self.num_actions)
                return e_greedy
            else:
                return self.table.select_greedy_action(state)
        else:
            return self.table.select_greedy_action(state)

    def update(self, state, action, reward, next_state, terminal):
        self.step_count += 1
        self.global_dict[AgentMonitor.Q_GLOBAL_STEPS] += 1

        if not self.testing:


            if terminal:
                self.data_dict['returns'].append(0)
                for i in range(len(self.data_dict['states'])):
                    new_val = self.data_dict['returns'][-i - 2] = self.data_dict['rewards'][-i - 1] + \
                                                        self.discounted_factor * self.data_dict['returns'][-i - 1]

                    cur_val = self.table.get_q_values(self.data_dict['actions'][-i - 1],
                                                      self.data_dict['states'][-i - 1])

                    new_val = (cur_val * self.eps_count + new_val)/(self.eps_count + 1)

                    self.table.set_q_values(self.data_dict['actions'][-i - 1],
                                            self.data_dict['states'][-i - 1],
                                            new_val)

            self.current_epsilon = self.epsilon_annealer.anneal(self.global_dict[AgentMonitor.Q_GLOBAL_STEPS])


if __name__ == '__main__':
    print([0] * 0)

