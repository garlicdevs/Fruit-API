from fruit.buffers.table import TLOLookupTable, LinearLookupTable
from fruit.learners.base import Learner
import numpy as np

from fruit.monitor.monitor import AgentMonitor
from fruit.utils.annealer import Annealer

table = None


class MOQLearner(Learner):
    def __init__(self, agent, name, environment, network, global_dict, report_frequency,
                 batch_size=5, discounted_factor=0.9, learning_rate=0.9, traces_factor=0.9,
                 epsilon_annealing_start=0.9, epsilon_annealing_end=0,
                 load_model_path=None, thresholds=None, target_reward=None, is_linear=False,
                 using_e_greedy=True, async_update_steps=1,
                 ):
        super().__init__(agent=agent, name=name, environment=environment, network=network, global_dict=global_dict,
                         report_frequency=report_frequency)

        self.load_model_path = load_model_path
        self.target_reward = target_reward
        self.is_linear = is_linear
        self.discounted_factor = discounted_factor
        self.traces_factor = traces_factor
        self.using_e_greedy = using_e_greedy
        self.async_update_steps = async_update_steps

        self.num_of_objectives = environment.get_number_of_objectives()
        self.init_q_values = [0.] * self.num_of_objectives
        if thresholds is None:
            if not is_linear:
                self.thresholds = [0.] * (self.num_of_objectives - 1)
            else:
                self.thresholds = [1./self.num_of_objectives] * self.num_of_objectives
        else:
            self.thresholds = thresholds

        global table
        with global_dict[AgentMonitor.Q_LOCK]:
            if table is None:
                if not is_linear:
                    table = TLOLookupTable(environment=environment, init_value=0., thresholds=self.thresholds)
                else:
                    table = LinearLookupTable(environment=environment, init_value=0., thresholds=self.thresholds)

        self.table = table
        self.batch_size = batch_size
        self.epsilon_annealer = Annealer(epsilon_annealing_start, epsilon_annealing_end, self.agent.max_training_steps)
        self.current_learning_rate = learning_rate
        self.current_epsilon = epsilon_annealing_start
        self.converged = False
        if self.load_model_path is not None:
            self.load_model()

    @staticmethod
    def get_default_number_of_learners():
        return 1

    def load_model(self):
        self.table.load_value_function(self.load_model_path)
        print("Load values:")
        self.table.print_values()

    def save_model(self, file_name):
        print("Save values:")
        self.table.print_values()
        self.table.save_value_function(file_name)

    def get_action(self, state):
        if self.using_e_greedy:
            if np.random.uniform(0, 1) <= self.current_epsilon:
                e_greedy = np.random.randint(self.num_actions)
                return e_greedy
            else:
                return self.table.select_greedy_action(state)
        else:
            return self.table.select_greedy_action(state)

    def report(self, reward):
        print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:',
              self.environment.get_current_steps(), 'Step count:', self.step_count, 'Learning rate:',
              self.current_learning_rate, 'Epsilon:', self.current_epsilon, 'Thresholds:', self.thresholds)

        # Testing purpose
        if self.target_reward is not None and self.thresholds is not None:
            backup_epsilon = self.current_epsilon
            self.current_epsilon = 0
            greedy_reward = self.run_episode()
            self.global_dict[AgentMonitor.Q_ADD_REWARD](greedy_reward, self.environment.get_current_steps())
            self.current_epsilon = backup_epsilon
            converged = True
            for i in range(len(greedy_reward)):
                if greedy_reward[i] != self.target_reward[i]:
                    converged = False
                    break
            if converged:
                print("Converged")
                self.converged = True

    def update(self, state, action, reward, next_state, terminal):
        self.step_count += 1
        self.global_dict['counter'] += 1

        if not self.testing:
            if self.step_count % self.async_update_steps == 0:
                if not terminal:
                    greedy = self.get_action(state)
                    self.table.calculate_td_errors(action, state, greedy, next_state, self.discounted_factor, reward)
                else:
                    self.table.calculate_terminal_td_errors(action, state, self.discounted_factor, reward)
                self.table.update_td_errors(action, state, 1.0, self.current_learning_rate)

                self.current_epsilon = self.epsilon_annealer.anneal(self.global_dict[AgentMonitor.Q_GLOBAL_STEPS])
