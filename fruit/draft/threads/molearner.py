import numpy as np
import threading
from fruit.utils.annealer import Annealer
from fruit.buffers.buffer import StateBuffer


class MOBaseLearner(object):
    def __init__(self, num_of_objs):
        self.num_of_objs = num_of_objs

    def reset(self):
        pass

    def get_action(self, state):
        return NotImplemented

    def step(self, environment_step_fn, action):
        return environment_step_fn(action)

    def update(self, state, action, reward, next_state, terminal):
        return NotImplemented

    def episode_end(self):
        pass

    def run_episode(self, environment):

        environment.reset()
        self.reset()
        total_reward = [0] * self.num_of_objs

        state = environment.get_state()

        terminal = False
        while not terminal:

            action = self.get_action(state)

            reward = self.step(environment.step, action)

            total_reward = np.add(total_reward, reward)

            next_state = environment.get_state()

            terminal = environment.is_terminal()

            self.update(state, action, reward, next_state, terminal)

            state = next_state

            # environment.render()

        self.episode_end()
        return total_reward


class MOBaseThreadLearner(threading.Thread, MOBaseLearner):
    def __init__(self, agent, name, environment, global_dict, num_of_objs=1, async_update_steps=5,
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=10000, global_epsilon_annealing=True, report_frequency=1, gamma=0.9,
                 traces_factor=0.9, batch_size=5, load_model_path=None,
                 lookup_table=None, thresholds=None, target_reward=None, is_linear=False):
        MOBaseLearner.__init__(self, num_of_objs)
        threading.Thread.__init__(self)

        range, is_range = environment.get_action_space().get_range()
        if not is_range:
            raise ValueError("Does not support this type of action space")

        self.using_e_greedy = using_e_greedy
        if using_e_greedy:
            end_rand = np.random.choice(epsilon_annealing_choices, p=epsilon_annealing_probabilities)
            self.epsilon_annealer = Annealer(epsilon_annealing_start, end_rand, epsilon_annealing_steps)

        self.current_epsilon = epsilon_annealing_start
        self.step_count = 0
        self.eps_count = 0
        self.environment = environment
        self.name = name
        self.agent = agent
        self.gamma = gamma
        self.traces_factor = traces_factor
        self.batch_size = batch_size
        self.load_model_path = load_model_path

        self.num_actions = len(range)

        self.async_update_step = async_update_steps
        self.global_dict = global_dict
        self.global_epsilon_annealing = global_epsilon_annealing

        self.report_frequency = report_frequency

        self.minibatch_vars = {}
        self.reset_minibatch()

        self.testing = False
        self.target_reward = target_reward
        self.is_linear = is_linear

        self.thresholds = thresholds
        if self.thresholds is None:
            self.thresholds = [0] * (self.num_of_objs-1)

        self.pareto_solutions = self.environment.get_pareto_solutions()
        # if self.pareto_solutions is not None:
        #    for i in range(len(self.pareto_solutions)):

        self.table = lookup_table
        self.table.set_threshold(self.thresholds)
        if self.load_model_path is not None:
            self.agent.load_model()

        self.alpha = self.agent.get_current_learning_rate()

    def reset(self):
        self.testing = self.agent.is_testing_mode

        self.reset_minibatch()

        # self.environment.render()

    def run(self):
        while not self.global_dict['done']:
            reward = self.run_episode(self.environment)
            self.eps_count += 1
            if self.target_reward is None:
                self.global_dict['add_reward'](reward, self.environment.get_current_steps())

            if self.eps_count % self.report_frequency == 0:
                current_epsilon = ''
                if self.using_e_greedy:
                    current_epsilon = 'Current epsilon: {0}'.format(self.current_epsilon)
                print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:', self.environment.get_current_steps(),
                      'Step count:', self.step_count, current_epsilon)

                # Testing purpose
                if self.target_reward is not None and self.thresholds is not None:
                    backup_epsilon = self.current_epsilon
                    self.current_epsilon = 0
                    greedy_reward = self.run_episode(self.environment)
                    self.global_dict['add_reward'](greedy_reward, self.environment.get_current_steps())
                    self.current_epsilon = backup_epsilon
                    converged = True
                    for i in range(len(greedy_reward)):
                        if greedy_reward[i] != self.target_reward[i]:
                            converged = False
                            break
                    if converged:
                        print("Converged")
                        self.agent.converged = True

            if not self.testing:
                print(self.current_epsilon)
                self.anneal_epsilon()

    def update(self, *args, **kwargs):
        return NotImplemented

    def anneal_epsilon(self):
        if self.using_e_greedy:
            anneal_step = self.global_dict['counter'] if self.global_epsilon_annealing else self.step_count
            self.current_epsilon = self.epsilon_annealer.anneal_to(anneal_step)

    def get_action(self, state):
        if self.using_e_greedy:
            # print(self.table.select_greedy_action(state))
            if np.random.uniform(0, 1) <= self.current_epsilon:
                e_greedy = np.random.randint(self.num_actions)
                return e_greedy
            else:
                return self.table.select_greedy_action(state)
        else:
            return self.table.select_greedy_action(state)

    def reset_minibatch(self):
        pass


class MOQWorker(MOBaseThreadLearner):
    def __init__(self, agent, name, environment, global_dict,
                 async_update_steps=1, num_of_objs=1,
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000, report_frequency=1, global_epsilon_annealing=True, gamma=0.9,
                 traces_factor=0.9, batch_size=5, load_model_path=None, lookup_table=None, thresholds=None,
                 target_reward=None, is_linear=False):

        super().__init__(agent, name, environment, global_dict,
                         num_of_objs=num_of_objs, async_update_steps=async_update_steps,
                         using_e_greedy=using_e_greedy, epsilon_annealing_start=epsilon_annealing_start,
                         epsilon_annealing_choices=epsilon_annealing_choices,
                         epsilon_annealing_probabilities=epsilon_annealing_probabilities,
                         epsilon_annealing_steps=epsilon_annealing_steps,
                         global_epsilon_annealing=global_epsilon_annealing,
                         report_frequency=report_frequency,
                         gamma=gamma, traces_factor=traces_factor, batch_size=batch_size,
                         load_model_path=load_model_path, lookup_table=lookup_table,
                         thresholds=thresholds, target_reward=target_reward, is_linear=is_linear)

    def update(self, state, action, reward, next_state, terminal):

        self.step_count += 1
        self.global_dict['counter'] += 1

        if not self.testing:
            if self.step_count % self.async_update_step == 0:
                if not terminal:
                    greedy = self.get_action(state)
                    self.table.calculate_td_errors(action, state, greedy, next_state, self.gamma, reward)
                else:
                    self.table.calculate_terminal_td_errors(action, state, self.gamma, reward)
                self.table.update(action, state, 1.0, self.alpha)


class MOExpReplayBaseThreadLearner(threading.Thread, MOBaseLearner):
    def __init__(self, agent, name, environment, network, global_dict, async_update_steps=1, reward_clip_vals=None,
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=100000, global_epsilon_annealing=True, report_frequency=1):
        MOBaseLearner.__init__(self, network.get_config().get_num_of_objectives())
        threading.Thread.__init__(self)

        range, is_range = environment.get_action_space().get_range()
        if not is_range:
            raise ValueError("Does not support this type of action space")

        self.using_e_greedy = using_e_greedy
        if using_e_greedy:
            end_rand = np.random.choice(epsilon_annealing_choices, p=epsilon_annealing_probabilities)
            self.epsilon_annealer = Annealer(epsilon_annealing_start, end_rand, epsilon_annealing_steps)

        self.current_epsilon = epsilon_annealing_start
        self.step_count = 0
        self.eps_count = 0
        self.environment = environment
        self.reward_clip_vals = reward_clip_vals
        self.name = name
        self.agent = agent

        self.num_actions = len(range)

        self.network = network
        self.config = network.network_config
        self.history_length = self.config.get_history_length()
        if self.history_length > 1:
            self.frame_buffer = StateBuffer([1] + self.config.get_input_shape())

        self.async_update_step = async_update_steps
        self.global_dict = global_dict
        self.global_epsilon_annealing = global_epsilon_annealing

        self.report_frequency = report_frequency

        self.minibatch_vars = {}
        self.reset_minibatch()

        self.testing = False

    def reset(self):

        self.testing = self.agent.is_testing_mode

        self.reset_minibatch()

        self.network.reset_network()

        if self.history_length > 1:
            self.frame_buffer.reset()

        state = self.environment.get_state()

        if self.history_length > 1:
            for _ in range(self.history_length):
                self.frame_buffer.add_state(state)

    def run(self):
        while not self.global_dict['done']:
            reward = self.run_episode(self.environment)
            self.eps_count += 1
            #self.global_dict['add_reward'](reward, self.environment.get_current_steps())

            if self.eps_count % self.report_frequency == 0:
                current_epsilon = ''
                if self.using_e_greedy:
                    current_epsilon = 'Current epsilon: {0}'.format(self.current_epsilon)
                print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:', self.environment.get_current_steps(),
                      'Step count:', self.step_count, current_epsilon)

                # Testing purpose
                backup_epsilon = self.current_epsilon
                self.current_epsilon = 0
                greedy_reward = self.run_episode(self.environment)

                print("Greedy reward:", greedy_reward)

                self.global_dict['add_reward'](greedy_reward, self.environment.get_current_steps())
                self.current_epsilon = backup_epsilon

    def update(self, *args, **kwargs):
        return NotImplemented

    def anneal_epsilon(self):
        if self.using_e_greedy:
            anneal_step = self.global_dict['counter'] if self.global_epsilon_annealing else self.step_count
            self.current_epsilon = self.epsilon_annealer.anneal_to(anneal_step)

    def get_action(self, state):
        if self.using_e_greedy:
            if np.random.uniform(0, 1) <= self.current_epsilon:
                e_greedy = np.random.randint(self.num_actions)
                return e_greedy
            else:
                if self.history_length > 1:
                    return self.network.get_output(self.frame_buffer.get_buffer_add_state(state))
                else:
                    return self.network.get_output(state)
        else:
            if self.history_length > 1:
                return self.network.get_output(self.frame_buffer.get_buffer_add_state(state))
            else:
                return self.network.get_output(state)

    def reset_minibatch(self):
        pass


class MOExpReplayWorker(MOExpReplayBaseThreadLearner):
    def __init__(self, agent, name, environment, network, global_dict, replay, batch_size=32, warmup_steps = 50000,
                 async_update_steps=5, reward_clip_vals=[-1, 1],
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000000, global_epsilon_annealing=True, report_frequency=1):
        super().__init__(agent, name, environment, network, global_dict, async_update_steps, reward_clip_vals,
                 using_e_greedy, epsilon_annealing_start, epsilon_annealing_choices,
                 epsilon_annealing_probabilities,
                 epsilon_annealing_steps, global_epsilon_annealing, report_frequency)
        self.replay = replay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

    def update(self, state, action, reward, next_state, terminal):

        if self.history_length > 1:
            self.frame_buffer.add_state(state)

        if self.reward_clip_vals is not None:
            reward = np.clip(reward, *self.reward_clip_vals)

        if not self.testing:
            if self.history_length > 1:
                self.replay.append(self.frame_buffer.get_buffer()[0], action, reward,
                                   self.frame_buffer.get_buffer_add_state(next_state)[0], terminal)
            else:
                if isinstance(state, list):
                    self.replay.append(state, action, reward, next_state, terminal)
                else:
                    self.replay.append([state], action, reward, [next_state], terminal)

        self.step_count += 1
        self.global_dict['counter'] += 1

        if self.step_count < self.warmup_steps:
            return

        if not self.testing:
            if self.step_count % self.async_update_step == 0:
                summaries = self.global_dict['write_summaries_this_step']
                s,a,r,n,t = self.replay.get_mini_batch(batch_size=self.batch_size)
                self.agent.anneal_learning_rate(self.global_dict['counter'])
                if summaries:
                    self.global_dict['write_summaries_this_step'] = False
                    summary = self.network.train_network(s, a, r, n, t, self.agent.current_learning_rate,
                                                         global_step=self.global_dict['counter'], summaries=True)
                    self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                else:
                    self.network.train_network(s, a, r, n, t, self.agent.current_learning_rate,
                                               global_step=self.global_dict['counter'], summaries=False)

            self.anneal_epsilon()