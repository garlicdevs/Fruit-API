from fruit.learners.mo_q_learning import MOQLearner
from fruit.monitor.monitor import AgentMonitor


class MCLearner(MOQLearner):
    def __init__(self, agent, name, environment, network, global_dict, report_frequency,
                 discounted_factor=0.9, learning_rate=0.9,
                 epsilon_annealing_start=0.9, epsilon_annealing_end=0,
                 load_model_path=None, target_reward=None, using_e_greedy=True):
        super().__init__(agent=agent, name=name, environment=environment, network=network, global_dict=global_dict,
                         report_frequency=report_frequency, discounted_factor=discounted_factor,
                         learning_rate=learning_rate,
                         epsilon_annealing_start=epsilon_annealing_start, epsilon_annealing_end=epsilon_annealing_end,
                         load_model_path=load_model_path, thresholds=None, target_reward=target_reward, is_linear=True,
                         using_e_greedy=using_e_greedy, async_update_steps=1)

    def reset(self):
        super().reset()
        self.data_dict['states'] = []
        self.data_dict['actions'] = []
        self.data_dict['rewards'] = []
        self.data_dict['returns'] = []

    def report(self, reward):
        print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:',
              self.environment.get_current_steps(), 'Step count:', self.step_count, 'Learning rate:',
              self.current_learning_rate, 'Epsilon:', self.current_epsilon, 'Thresholds:', self.thresholds)
        print(self.table.print_values())

    def update(self, state, action, reward, next_state, terminal):
        self.step_count += 1
        self.global_dict['counter'] += 1

        if not self.testing:
            self.data_dict['states'].append(state)
            self.data_dict['actions'].append(action)
            self.data_dict['rewards'].append(reward)
            self.data_dict['returns'].append(0)
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