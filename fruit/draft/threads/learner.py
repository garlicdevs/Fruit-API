import threading

from fruit.buffers.replay import AsyncExperience
from fruit.utils.annealer import Annealer
from fruit.buffers.buffer import StateBuffer
from copy import deepcopy
from sys import platform as _platform
from fruit.utils.image import *


class BaseLearner(object):
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
        total_reward = 0

        state = environment.get_state()

        terminal = False
        while not terminal:
            action = self.get_action(state)

            reward = self.step(environment.step, action)

            total_reward += reward

            next_state = environment.get_state()

            terminal = environment.is_terminal()

            self.update(state, action, reward, next_state, terminal)

            state = next_state

        self.episode_end()
        return total_reward


class MODQNBaseLearner(object):
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
        total_reward = [0] * 3

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

        self.episode_end()
        return total_reward


class MOBaseLearner(object):
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
        objs = environment.get_num_of_objectives()
        self.reset()
        total_reward = [0] * objs

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

        self.episode_end()
        return total_reward


class MABaseLearner(object):
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
        total_reward = 0

        state = environment.get_state()

        num_of_agents = environment.get_num_of_agents()

        terminal = False
        while not terminal:
            action = self.get_action(state)

            reward = self.step(environment.step, action)

            if type(reward) is list:
                for i in range(num_of_agents):
                    total_reward = total_reward + reward[i]
            else:
                total_reward = total_reward + reward

            next_state = environment.get_state()

            terminal = environment.is_terminal()

            self.update(state, action, reward, next_state, terminal)

            state = next_state

        self.episode_end()
        return total_reward


class MapMABaseLearner(object):
    def reset(self):
        pass

    def get_action(self, state, map):
        return NotImplemented

    def step(self, environment_step_fn, action):
        return environment_step_fn(action)

    def update(self, state, action, reward, next_state, terminal, map):
        return NotImplemented

    def episode_end(self):
        pass

    def run_episode(self, environment):

        environment.reset()
        self.reset()
        total_reward = 0

        state = environment.get_state()
        map = environment.get_map()

        num_of_agents = environment.get_num_of_agents()

        terminal = False

        while not terminal:
            action = self.get_action(state, map)

            reward = self.step(environment.step, action)

            if type(reward) is list:
                for i in range(num_of_agents):
                    total_reward = total_reward + reward[i]
            else:
                total_reward = total_reward + reward

            next_state = environment.get_state()
            map = environment.get_map()

            terminal = environment.is_terminal()

            self.update(state, action, reward, next_state, terminal, map)

            state = next_state

        self.episode_end()
        return total_reward


class NIPSBaseLearner(object):

    def reset(self):
        pass

    def get_action(self, state, map_data):
        return NotImplemented

    def step(self, environment_step_fn, action):
        return environment_step_fn(action)

    def update(self, state, action, reward, next_state, terminal, map_data):
        return NotImplemented

    def episode_end(self):
        pass

    def run_episode(self, environment):

        environment.reset()
        self.reset()
        total_reward = 0

        state = environment.get_state()
        map_data = environment.get_map()

        # save_grayscale_image(map, full_path="/Users/alpha/Desktop/images/image" + str(0) + ".png")

        terminal = False
        while not terminal:
            action = self.get_action(state, map_data)

            reward = self.step(environment.step, action)

            # total_reward = total_reward + reward[0] + reward[1]
        
            total_reward = total_reward + reward[0]
            oxy_low = reward[3]
            lives = reward[4]
            #total_reward = np.add(total_reward, reward)

            next_state = environment.get_state()

            terminal = environment.is_terminal()

            next_map_data = environment.get_map()

            # self.update(state, action, reward[2], next_state, terminal, map_data)
            # print(reward[0], reward[1], reward[2])
            if oxy_low == 1:
                update_reward = reward[2]
            else:
                update_reward = reward[0] + reward[1]

            self.update(state, action, update_reward, next_state, terminal, map_data)

            state = next_state

            map_data = next_map_data

        self.episode_end()
        return total_reward


class MANIPSBaseLearner(object):
    def reset(self):
        pass

    def get_action(self, state, map_data, map_data_2):
        return NotImplemented

    def step(self, environment_step_fn, action):
        return environment_step_fn(action)

    def update(self, state, action, reward, next_state, terminal, map_data, map_data_2):
        return NotImplemented

    def episode_end(self):
        pass

    def run_episode(self, environment):

        environment.reset()
        self.reset()
        total_reward = 0

        state = environment.get_state()
        map_data = environment.get_map(0)
        map_data_2 = environment.get_map(1)

        reward = self.step(environment.step, [2, 2])
        reward = self.step(environment.step, [2, 2])

        terminal = False
        while not terminal:
            action = self.get_action(state, map_data, map_data_2)

            reward = self.step(environment.step, action)

            total_reward = total_reward + reward[0] + reward[1]

            next_state = environment.get_state()

            terminal = environment.is_terminal()

            next_map_data = environment.get_map(0)
            next_map_data_2 = environment.get_map(1)

            self.update(state, action, reward[2], next_state, terminal, map_data, map_data_2)

            state = next_state

            map_data = next_map_data
            map_data_2 = next_map_data_2

        self.episode_end()
        return total_reward


class BaseThreadLearner(threading.Thread, BaseLearner):
    def __init__(self, agent, name, environment, network, global_dict, async_update_steps=5,
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000000, global_epsilon_annealing=True, report_frequency=1):
        super().__init__()

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

        self.num_actions = len(range)

        self.network = network
        self.config = network.network_config
        self.history_length = self.config.get_history_length()
        if self.history_length > 1:
            self.frame_buffer = StateBuffer(self.config.get_input_shape(), history_length=self.history_length)

        self.async_update_step = async_update_steps
        self.global_dict = global_dict
        self.global_epsilon_annealing = global_epsilon_annealing

        self.report_frequency = report_frequency

        self.testing = False

        self.initialize()

    def initialize(self):
        pass

    def reset(self):

        self.testing = self.agent.is_testing_mode

        self.network.reset_network()

        if self.history_length > 1:
            self.frame_buffer.reset()

        state = self.environment.get_state()

        if self.history_length > 1:
            for _ in range(self.history_length):
                self.frame_buffer.add_state(state)

    def _run(self):
        reward = self.run_episode(self.environment)
        self.eps_count += 1
        self.global_dict['add_reward'](reward, self.environment.get_current_steps())

        if self.eps_count % self.report_frequency == 0:
            current_epsilon = ''
            if self.using_e_greedy:
                current_epsilon = 'Current epsilon: {0}'.format(self.current_epsilon)
            print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:',
                  self.environment.get_current_steps(), 'Step count:', self.step_count, current_epsilon)

    def run(self):
        if self.environment.is_render and _platform == "darwin":
            self._run()
        else:
            while not self.global_dict['done']:
                self._run()

    def update(self, *args, **kwargs):
        return NotImplemented

    def anneal_epsilon(self):
        if self.using_e_greedy:
            anneal_step = self.global_dict['counter'] if self.global_epsilon_annealing else self.step_count
            self.current_epsilon = self.epsilon_annealer.anneal(anneal_step)

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


class MOBaseThreadLearner(threading.Thread, MOBaseLearner):
    def __init__(self, agent, name, environment, network, global_dict, async_update_steps=5, reward_clip_vals=[-1, 1],
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000000, global_epsilon_annealing=True, report_frequency=1, thread_id=-1,
                 total_weights=-1):
        super().__init__()

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
        self.total_weights = total_weights
        self.thread_id = thread_id
        self.thread_id = self.thread_id % self.total_weights
        #print(self.thread_id, self.total_weights)

        self.num_actions = len(range)

        self.network = network
        self.config = network.network_config
        self.history_length = self.config.get_history_length()
        if self.history_length > 1:
            self.frame_buffer = StateBuffer(self.config.get_input_shape(), history_length=self.history_length)

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

    def _run(self):
        reward = self.run_episode(self.environment)
        self.eps_count += 1
        self.global_dict['add_reward'](reward, self.environment.get_current_steps())

        if self.eps_count % self.report_frequency == 0:
            current_epsilon = ''
            if self.using_e_greedy:
                current_epsilon = 'Current epsilon: {0}'.format(self.current_epsilon)
            print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:',
                  self.environment.get_current_steps(),
                  'Step count:', self.step_count, current_epsilon)

    def run(self):
        if self.environment.is_render and _platform == "darwin":
            self._run()
        else:
            while not self.global_dict['done']:
                self._run()

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


class MABaseThreadLearner(threading.Thread, MABaseLearner):
    def __init__(self, agent, name, environment, network, global_dict, async_update_steps=5, reward_clip_vals=[-1, 1],
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000000, global_epsilon_annealing=True, report_frequency=1):
        super().__init__()

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
            self.frame_buffer = StateBuffer(self.config.get_input_shape(), history_length=self.history_length)

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

    def _run(self):
        reward = self.run_episode(self.environment)
        self.eps_count += 1
        self.global_dict['add_reward'](reward, self.environment.get_current_steps())

        if self.eps_count % self.report_frequency == 0:
            current_epsilon = ''
            if self.using_e_greedy:
                current_epsilon = 'Current epsilon: {0}'.format(self.current_epsilon)
            print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:',
                  self.environment.get_current_steps(),
                  'Step count:', self.step_count, current_epsilon)

    def run(self):
        if self.environment.is_render and _platform == "darwin":
            self._run()
        else:
            while not self.global_dict['done']:
                self._run()

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


class MapMABaseThreadLearner(threading.Thread, MapMABaseLearner):
    def __init__(self, agent, name, environment, network, global_dict, async_update_steps=5, reward_clip_vals=[-1, 1],
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000000, global_epsilon_annealing=True, report_frequency=1):
        super().__init__()

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
            self.frame_buffer = StateBuffer(self.config.get_input_shape(), history_length=self.history_length)
            self.map_buffer = StateBuffer(self.config.get_input_shape(), history_length=self.history_length)

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
            self.map_buffer.reset()

        state = self.environment.get_state()
        map = self.environment.get_map()

        if self.history_length > 1:
            for _ in range(self.history_length):
                self.frame_buffer.add_state(state)
                self.map_buffer.add_state(map)

    def _run(self):
        reward = self.run_episode(self.environment)
        self.eps_count += 1
        self.global_dict['add_reward'](reward, self.environment.get_current_steps())

        if self.eps_count % self.report_frequency == 0:
            current_epsilon = ''
            if self.using_e_greedy:
                current_epsilon = 'Current epsilon: {0}'.format(self.current_epsilon)
            print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:',
                  self.environment.get_current_steps(),
                  'Step count:', self.step_count, current_epsilon)

    def run(self):
        if self.environment.is_render and _platform == "darwin":
            self._run()
        else:
            while not self.global_dict['done']:
                self._run()

    def update(self, *args, **kwargs):
        return NotImplemented

    def anneal_epsilon(self):
        if self.using_e_greedy:
            anneal_step = self.global_dict['counter'] if self.global_epsilon_annealing else self.step_count
            self.current_epsilon = self.epsilon_annealer.anneal_to(anneal_step)

    def get_action(self, state, map):
        if self.using_e_greedy:
            if np.random.uniform(0, 1) <= self.current_epsilon:
                e_greedy = np.random.randint(self.num_actions)
                return e_greedy
            else:
                if self.history_length > 1:
                    return self.network.get_output(self.frame_buffer.get_buffer_add_state(state),
                                                   self.map_buffer.get_buffer_add_state(map))
                else:
                    return self.network.get_output(state, map)
        else:
            if self.history_length > 1:
                return self.network.get_output(self.frame_buffer.get_buffer_add_state(state),
                                               self.frame_buffer.get_buffer_add_state(map))
            else:
                return self.network.get_output(state, map)

    def reset_minibatch(self):
        pass


class NIPSBaseThreadLearner(threading.Thread, NIPSBaseLearner):
    def __init__(self, agent, name, environment, network, global_dict, async_update_steps=5, reward_clip_vals=[-1, 1],
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000000, global_epsilon_annealing=True, report_frequency=1):
        super().__init__()

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
            self.frame_buffer = StateBuffer(self.config.get_input_shape(), history_length=self.history_length)
            self.map_buffer = StateBuffer(self.config.get_input_shape(), history_length=self.history_length)

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
            self.map_buffer.reset()

        state = self.environment.get_state()
        map = self.environment.get_map()

        if self.history_length > 1:
            for _ in range(self.history_length):
                self.frame_buffer.add_state(state)
                self.frame_buffer.add_state(map)

    def _run(self):
        reward = self.run_episode(self.environment)
        self.eps_count += 1
        self.global_dict['add_reward'](reward, self.environment.get_current_steps())

        if self.eps_count % self.report_frequency == 0:
            current_epsilon = ''
            if self.using_e_greedy:
                current_epsilon = 'Current epsilon: {0}'.format(self.current_epsilon)
            print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:',
                  self.environment.get_current_steps(),
                  'Step count:', self.step_count, current_epsilon)

    def run(self):
        if self.environment.is_render and _platform == "darwin":
            self._run()
        else:
            while not self.global_dict['done']:
                self._run()

    def update(self, *args, **kwargs):
        return NotImplemented

    def anneal_epsilon(self):
        if self.using_e_greedy:
            anneal_step = self.global_dict['counter'] if self.global_epsilon_annealing else self.step_count
            self.current_epsilon = self.epsilon_annealer.anneal_to(anneal_step)

    def get_action(self, state, map_data):
        if self.using_e_greedy:
            if np.random.uniform(0, 1) <= self.current_epsilon:
                e_greedy = np.random.randint(self.num_actions)
                return e_greedy
            else:
                if self.history_length > 1:
                    return self.network.get_output(self.frame_buffer.get_buffer_add_state(state),
                                                   self.map_buffer.get_buffer_add_state(map_data))
                else:
                    return self.network.get_output(state, map_data)
        else:
            if self.history_length > 1:
                return self.network.get_output(self.frame_buffer.get_buffer_add_state(state),
                                               self.map_buffer.get_buffer_add_state(map_data))
            else:
                return self.network.get_output(state, map_data)

    def reset_minibatch(self):
        pass


class MANIPSBaseThreadLearner(threading.Thread, MANIPSBaseLearner):
    def __init__(self, agent, name, environment, network, global_dict, async_update_steps=5, reward_clip_vals=[-1, 1],
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000000, global_epsilon_annealing=True, report_frequency=1):
        super().__init__()

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
            self.frame_buffer = StateBuffer(self.config.get_input_shape(), history_length=self.history_length)
            self.map_buffer = StateBuffer(self.config.get_input_shape(), history_length=self.history_length)
            self.map_buffer_2 = StateBuffer(self.config.get_input_shape(), history_length=self.history_length)

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
            self.map_buffer.reset()
            self.map_buffer_2.reset()

        state = self.environment.get_state()
        map = self.environment.get_map(0)
        map2 = self.environment.get_map(1)

        if self.history_length > 1:
            for _ in range(self.history_length):
                self.frame_buffer.add_state(state)
                self.frame_buffer.add_state(map)
                self.frame_buffer.add_state(map2)

    def _run(self):
        reward = self.run_episode(self.environment)
        self.eps_count += 1
        self.global_dict['add_reward'](reward, self.environment.get_current_steps())

        if self.eps_count % self.report_frequency == 0:
            current_epsilon = ''
            if self.using_e_greedy:
                current_epsilon = 'Current epsilon: {0}'.format(self.current_epsilon)
            print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:',
                  self.environment.get_current_steps(),
                  'Step count:', self.step_count, current_epsilon)

    def run(self):
        if self.environment.is_render and _platform == "darwin":
            self._run()
        else:
            while not self.global_dict['done']:
                self._run()

    def update(self, *args, **kwargs):
        return NotImplemented

    def anneal_epsilon(self):
        if self.using_e_greedy:
            anneal_step = self.global_dict['counter'] if self.global_epsilon_annealing else self.step_count
            self.current_epsilon = self.epsilon_annealer.anneal_to(anneal_step)

    def get_action(self, state, map_data, map_data_2):
        if self.using_e_greedy:
            if np.random.uniform(0, 1) <= self.current_epsilon:
                e_greedy = np.random.randint(self.num_actions)
                return e_greedy
            else:
                if self.history_length > 1:
                    return self.network.get_output(self.frame_buffer.get_buffer_add_state(state),
                                                   self.map_buffer.get_buffer_add_state(map_data))
                else:
                    return self.network.get_output(state, map_data)
        else:
            if self.history_length > 1:
                return self.network.get_output(self.frame_buffer.get_buffer_add_state(state),
                                               self.map_buffer.get_buffer_add_state(map_data),
                                               self.map_buffer_2.get_buffer_add_state(map_data_2)
                                               )
            else:
                return self.network.get_output(state, map_data)

    def reset_minibatch(self):
        pass


class ExpReplayWorker(BaseThreadLearner):
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


class MODQNBaseThreadLearner(threading.Thread, MODQNBaseLearner):
    def __init__(self, agent, name, environment, network, global_dict, async_update_steps=5, reward_clip_vals=[-1, 1],
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000000, global_epsilon_annealing=True, report_frequency=1):
        super().__init__()

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
            self.frame_buffer = StateBuffer(self.config.get_input_shape(), history_length=self.history_length)

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

    def _run(self):
        reward = self.run_episode(self.environment)
        self.eps_count += 1
        self.global_dict['add_reward'](reward, self.environment.get_current_steps())

        if self.eps_count % self.report_frequency == 0:
            current_epsilon = ''
            if self.using_e_greedy:
                current_epsilon = 'Current epsilon: {0}'.format(self.current_epsilon)
            print(self.name, 'Episode Count:', self.eps_count, 'Episode reward:', reward, 'Steps:',
                  self.environment.get_current_steps(),
                  'Step count:', self.step_count, current_epsilon)

    def run(self):
        if self.environment.is_render and _platform == "darwin":
            self._run()
        else:
            while not self.global_dict['done']:
                self._run()

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


class MOExpReplayWorker(MODQNBaseThreadLearner):
    def __init__(self, agent, name, environment, network, global_dict, replay, batch_size=32, warmup_steps = 50000,
                 async_update_steps=5, reward_clip_vals=[-1, 1],
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000000, global_epsilon_annealing=True, report_frequency=1, weights=None,
                 id=0, num_of_threads=1):
        super().__init__(agent, name, environment, network, global_dict, async_update_steps, reward_clip_vals,
                 using_e_greedy, epsilon_annealing_start, epsilon_annealing_choices,
                 epsilon_annealing_probabilities,
                 epsilon_annealing_steps, global_epsilon_annealing, report_frequency)
        self.replay = replay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.weights = weights
        self.thread_id = id
        self.count = 0
        self.num_of_threads = num_of_threads

    def update(self, state, action, reward, next_state, terminal):

        reward = np.sum(np.multiply(reward, self.weights))

        if self.num_of_threads == 1:
            if terminal:
                self.count = self.count + 1
                self.count = self.count % self.num_of_threads
            round = self.count
        else:
            round = self.thread_id % len(self.weights)

        if self.history_length > 1:
            self.frame_buffer.add_state(state)

        if self.reward_clip_vals is not None:
            reward = np.clip(reward, *self.reward_clip_vals)

        if not self.testing:
            if self.history_length > 1:
                self.replay[round].append(self.frame_buffer.get_buffer()[0], action, reward,
                                          self.frame_buffer.get_buffer_add_state(next_state)[0], terminal)
            else:
                self.replay[round].append([state], action, reward, [next_state], terminal)

        self.step_count += 1
        self.global_dict['counter'] += 1

        if self.step_count < self.warmup_steps:
            return

        if not self.testing:
            if self.step_count % self.async_update_step == 0:
                summaries = self.global_dict['write_summaries_this_step']
                s,a,r,n,t = self.replay[round].get_mini_batch(batch_size=self.batch_size)
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


class PrioritizedExpReplayWorker(BaseThreadLearner):
    def __init__(self, agent, name, environment, network, global_dict, replay, batch_size=32, warmup_steps = 50000,
                 async_update_steps=5, reward_clip_vals=[-1, 1],
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000000, global_epsilon_annealing=True, report_frequency=1):
        super().__init__(agent, name, environment, network, global_dict, async_update_steps, reward_clip_vals,
                         using_e_greedy, epsilon_annealing_start, epsilon_annealing_choices,
                         epsilon_annealing_probabilities, epsilon_annealing_steps,
                         global_epsilon_annealing, report_frequency)
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
                self.replay.append([state], action, reward, [next_state], terminal)

        self.step_count += 1
        self.global_dict['counter'] += 1

        if self.step_count < self.warmup_steps:
            return

        if not self.testing:
            if self.step_count % self.async_update_step == 0:
                summaries = self.global_dict['write_summaries_this_step']
                s, a, r, n, t, e, w, p, mw = self.replay.get_mini_batch(batch_size=self.batch_size,
                                                                        current_beta=self.agent.current_beta)

                self.agent.anneal_learning_rate(self.global_dict['counter'])
                self.agent.anneal_beta(self.global_dict['counter'])
                if summaries:
                    self.global_dict['write_summaries_this_step'] = False
                    summary = self.network.train_network(s, a, r, n, t, self.agent.current_learning_rate,
                                                         global_step=self.global_dict['counter'], summaries=True,
                                                         other=w)
                    self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                else:
                    self.network.train_network(s, a, r, n, t, self.agent.current_learning_rate,
                                               global_step=self.global_dict['counter'], summaries=False,
                                               other=w)

                td_errors = self.network.network_config.get_td_errors(self.network.get_session(), s, a, r, n, t)
                if self.reward_clip_vals is not None:
                    td_errors = np.clip(td_errors, *self.reward_clip_vals)

                self.replay.update_mini_batch(e, td_errors)

                if self.step_count % 100000 == 0:
                    print('###################################################################')
                    print('TD Errors:', td_errors)
                    print('Beta:', self.agent.current_beta)
                    print('Mini Batches:', e)
                    print('Weights:', w)
                    print('Max Weight:', mw)
                    print('Probability:', p)
                    print('###################################################################')

            self.anneal_epsilon()


class MOA3CWorker(MOBaseThreadLearner):
    def update(self, state, action, reward, next_state, terminal):

        if self.history_length > 1:
            self.frame_buffer.add_state(state)

        if self.reward_clip_vals is not None:
            reward = np.clip(reward, *self.reward_clip_vals)

        if not self.testing:
            if self.history_length > 1:
                self.minibatch_accumulate(self.frame_buffer.get_buffer(), action,
                                          reward, self.frame_buffer.get_buffer_add_state(next_state), terminal)
            else:
                self.minibatch_accumulate([state], action, reward, [next_state], terminal)

        self.step_count += 1
        self.global_dict['counter'] += 1

        if not self.testing:
            if self.step_count % self.async_update_step == 0 or terminal:
                summaries = self.global_dict['write_summaries_this_step']
                self.agent.anneal_learning_rate(self.global_dict['counter'])
                if summaries:
                    self.global_dict['write_summaries_this_step'] = False
                    summary = self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                                         global_step=self.global_dict['counter'], summaries=True)
                    self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                else:
                    self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                               global_step=self.global_dict['counter'], summaries=False)
                self.reset_minibatch()

            self.anneal_epsilon()

    def minibatch_accumulate(self, state, action, reward, state_tp1, terminal):
        self.minibatch_vars['states'].append(state[0])
        self.minibatch_vars['actions'].append(action)
        self.minibatch_vars['rewards'].append(reward)
        self.minibatch_vars['state_tp1s'].append(state_tp1[0])
        self.minibatch_vars['terminals'].append(terminal)

    def reset_minibatch(self):
        self.minibatch_vars['states'] = []
        self.minibatch_vars['actions'] = []
        self.minibatch_vars['rewards'] = []
        self.minibatch_vars['state_tp1s'] = []
        self.minibatch_vars['terminals'] = []

    def get_minibatch_vars(self):
        return [self.minibatch_vars['states'],
                self.minibatch_vars['actions'],
                self.minibatch_vars['rewards'],
                self.minibatch_vars['state_tp1s'],
                self.minibatch_vars['terminals']]


class JairA3CWorker(MOBaseThreadLearner):

    def get_action(self, state):
        if self.using_e_greedy:
            if np.random.uniform(0, 1) <= self.current_epsilon:
                e_greedy = np.random.randint(self.num_actions)
                return e_greedy
            else:
                if self.history_length > 1:
                    return self.network.get_output_with_weight(self.frame_buffer.get_buffer_add_state(state), self.thread_id)
                else:
                    return self.network.get_output_with_weight(state, self.thread_id)
        else:
            if self.history_length > 1:
                return self.network.get_output_with_weight(self.frame_buffer.get_buffer_add_state(state), self.thread_id)
            else:
                return self.network.get_output_with_weight(state, self.thread_id)

    def update(self, state, action, reward, next_state, terminal):

        if self.history_length > 1:
            self.frame_buffer.add_state(state)

        if self.reward_clip_vals is not None:
            reward = np.clip(reward, *self.reward_clip_vals)

        if not self.testing:
            if self.history_length > 1:
                self.minibatch_accumulate(self.frame_buffer.get_buffer(), action,
                                          reward, self.frame_buffer.get_buffer_add_state(next_state), terminal)
            else:
                self.minibatch_accumulate([state], action, reward, [next_state], terminal)

        self.step_count += 1
        self.global_dict['counter'] += 1

        if not self.testing:
            if self.step_count % self.async_update_step == 0 or terminal:
                summaries = self.global_dict['write_summaries_this_step']
                self.agent.anneal_learning_rate(self.global_dict['counter'])
                if summaries:
                    self.global_dict['write_summaries_this_step'] = False
                    summary = self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                                         global_step=self.global_dict['counter'], summaries=True,
                                                         other=self.thread_id)
                    self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                else:
                    self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                               global_step=self.global_dict['counter'], summaries=False,
                                               other=self.thread_id)
                self.reset_minibatch()

            self.anneal_epsilon()

    def minibatch_accumulate(self, state, action, reward, state_tp1, terminal):
        self.minibatch_vars['states'].append(state[0])
        self.minibatch_vars['actions'].append(action)
        self.minibatch_vars['rewards'].append(reward)
        self.minibatch_vars['state_tp1s'].append(state_tp1[0])
        self.minibatch_vars['terminals'].append(terminal)

    def reset_minibatch(self):
        self.minibatch_vars['states'] = []
        self.minibatch_vars['actions'] = []
        self.minibatch_vars['rewards'] = []
        self.minibatch_vars['state_tp1s'] = []
        self.minibatch_vars['terminals'] = []

    def get_minibatch_vars(self):
        return [self.minibatch_vars['states'],
                self.minibatch_vars['actions'],
                self.minibatch_vars['rewards'],
                self.minibatch_vars['state_tp1s'],
                self.minibatch_vars['terminals']]


class MapMAA3CWorker(MapMABaseThreadLearner):
    def update(self, state, action, reward, next_state, terminal, map):

        num_of_agents = self.environment.get_num_of_agents()

        if self.history_length > 1:
            self.frame_buffer.add_state(state)
            self.map_buffer.add_state(map)

        if self.reward_clip_vals is not None:
            if type(reward) is list:
                for i in range(num_of_agents):
                    reward[i] = np.clip(reward[i], *self.reward_clip_vals)
            else:
                reward = np.clip(reward, *self.reward_clip_vals)

        if not self.testing:
            if self.history_length > 1:
                self.minibatch_accumulate(self.frame_buffer.get_buffer(), action,
                                          reward, self.frame_buffer.get_buffer_add_state(next_state), terminal,
                                          self.map_buffer.get_buffer())
            else:
                self.minibatch_accumulate([state], action, reward, [next_state], terminal, [map])

        self.step_count += 1
        self.global_dict['counter'] += 1

        if not self.testing:
            if self.step_count % self.async_update_step == 0 or terminal:
                summaries = self.global_dict['write_summaries_this_step']
                self.agent.anneal_learning_rate(self.global_dict['counter'])
                if summaries:
                    self.global_dict['write_summaries_this_step'] = False

                    summary = self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                                         global_step=self.global_dict['counter'], summaries=True)
                    self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                else:

                    self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                               global_step=self.global_dict['counter'], summaries=False)
                self.reset_minibatch()

            self.anneal_epsilon()

    def minibatch_accumulate(self, state, action, reward, state_tp1, terminal, map):
        self.minibatch_vars['states'].append(state[0])
        self.minibatch_vars['actions'].append(action)
        self.minibatch_vars['rewards'].append(reward)
        self.minibatch_vars['state_tp1s'].append(state_tp1[0])
        self.minibatch_vars['terminals'].append(terminal)
        self.minibatch_vars['map'].append(map[0])

    def reset_minibatch(self):
        self.minibatch_vars['states'] = []
        self.minibatch_vars['actions'] = []
        self.minibatch_vars['rewards'] = []
        self.minibatch_vars['state_tp1s'] = []
        self.minibatch_vars['terminals'] = []
        self.minibatch_vars['map'] = []

    def get_minibatch_vars(self):
        return [self.minibatch_vars['states'],
                self.minibatch_vars['actions'],
                self.minibatch_vars['rewards'],
                self.minibatch_vars['state_tp1s'],
                self.minibatch_vars['terminals'],
                self.minibatch_vars['map']]


class MAA3CWorker(MABaseThreadLearner):
    def update(self, state, action, reward, next_state, terminal):

        num_of_agents = self.environment.get_num_of_agents()

        if self.history_length > 1:
            self.frame_buffer.add_state(state)

        if self.reward_clip_vals is not None:
            if type(reward) is list:
                for i in range(num_of_agents):
                    reward[i] = np.clip(reward[i], *self.reward_clip_vals)
            else:
                reward = np.clip(reward, *self.reward_clip_vals)

        if not self.testing:
            if self.history_length > 1:
                self.minibatch_accumulate(self.frame_buffer.get_buffer(), action,
                                          reward, self.frame_buffer.get_buffer_add_state(next_state), terminal)
            else:
                self.minibatch_accumulate([state], action, reward, [next_state], terminal)

        self.step_count += 1
        self.global_dict['counter'] += 1

        if not self.testing:
            if self.step_count % self.async_update_step == 0 or terminal:
                summaries = self.global_dict['write_summaries_this_step']
                self.agent.anneal_learning_rate(self.global_dict['counter'])
                if summaries:
                    self.global_dict['write_summaries_this_step'] = False

                    summary = self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                                         global_step=self.global_dict['counter'], summaries=True)
                    self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                else:

                    self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                               global_step=self.global_dict['counter'], summaries=False)
                self.reset_minibatch()

            self.anneal_epsilon()

    def minibatch_accumulate(self, state, action, reward, state_tp1, terminal):
        self.minibatch_vars['states'].append(state[0])
        self.minibatch_vars['actions'].append(action)
        self.minibatch_vars['rewards'].append(reward)
        self.minibatch_vars['state_tp1s'].append(state_tp1[0])
        self.minibatch_vars['terminals'].append(terminal)

    def reset_minibatch(self):
        self.minibatch_vars['states'] = []
        self.minibatch_vars['actions'] = []
        self.minibatch_vars['rewards'] = []
        self.minibatch_vars['state_tp1s'] = []
        self.minibatch_vars['terminals'] = []

    def get_minibatch_vars(self):
        return [self.minibatch_vars['states'],
                self.minibatch_vars['actions'],
                self.minibatch_vars['rewards'],
                self.minibatch_vars['state_tp1s'],
                self.minibatch_vars['terminals']]


class RiverraidA3CWorker(BaseThreadLearner):

    def episode_end(self):
        # print("Epsiode ends")
        self.environment.processor.reset()

    def update(self, state, action, reward, next_state, terminal):

        gas = self.environment.processor.get_gas_percentage()
        # print("Gas:", gas)

        if reward == 80:
            # print("Dont shoot the gas tank")
            reward = -10
        if reward == 0:
            reward = self.environment.processor.get_intrinsic_reward()
        #    if reward != 0:
        #        print("Gas reward")

        # if reward != 0:
        #    print("Reward:", reward)

        if self.history_length > 1:
            self.frame_buffer.add_state(state)

        if self.reward_clip_vals is not None:
            reward = np.clip(reward, *self.reward_clip_vals)

        if not self.testing:
            if self.history_length > 1:
                self.minibatch_accumulate(self.frame_buffer.get_buffer(), action,
                                          reward, self.frame_buffer.get_buffer_add_state(next_state), terminal)
            else:
                self.minibatch_accumulate([state], action, reward, [next_state], terminal)

        self.step_count += 1
        self.global_dict['counter'] += 1

        if not self.testing:
            if self.step_count % self.async_update_step == 0 or terminal:
                summaries = self.global_dict['write_summaries_this_step']
                self.agent.anneal_learning_rate(self.global_dict['counter'])
                if summaries:
                    self.global_dict['write_summaries_this_step'] = False
                    summary = self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                                         global_step=self.global_dict['counter'], summaries=True)
                    self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                else:
                    self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                               global_step=self.global_dict['counter'], summaries=False)
                self.reset_minibatch()

            self.anneal_epsilon()

    def minibatch_accumulate(self, state, action, reward, state_tp1, terminal):
        self.minibatch_vars['states'].append(state[0])
        self.minibatch_vars['actions'].append(action)
        self.minibatch_vars['rewards'].append(reward)
        self.minibatch_vars['state_tp1s'].append(state_tp1[0])
        self.minibatch_vars['terminals'].append(terminal)

    def reset_minibatch(self):
        self.minibatch_vars['states'] = []
        self.minibatch_vars['actions'] = []
        self.minibatch_vars['rewards'] = []
        self.minibatch_vars['state_tp1s'] = []
        self.minibatch_vars['terminals'] = []

    def get_minibatch_vars(self):
        return [self.minibatch_vars['states'],
                self.minibatch_vars['actions'],
                self.minibatch_vars['rewards'],
                self.minibatch_vars['state_tp1s'],
                self.minibatch_vars['terminals']]


class NIPSA3CWorker(NIPSBaseThreadLearner):

    def update(self, state, action, reward, next_state, terminal, map_data):

        if self.history_length > 1:
            self.frame_buffer.add_state(state)
            self.map_buffer.add_state(map_data)

        if self.reward_clip_vals is not None:
            reward = np.clip(reward, *self.reward_clip_vals)

        if not self.testing:
            if self.history_length > 1:
                self.minibatch_accumulate(self.frame_buffer.get_buffer(), action,
                                          reward, self.frame_buffer.get_buffer_add_state(next_state), terminal,
                                          self.map_buffer.get_buffer())
            else:
                self.minibatch_accumulate([state], action, reward, [next_state], terminal, [map_data])

        self.step_count += 1
        self.global_dict['counter'] += 1

        if not self.testing:
            if self.step_count % self.async_update_step == 0 or terminal:
                summaries = self.global_dict['write_summaries_this_step']
                self.agent.anneal_learning_rate(self.global_dict['counter'])
                if summaries:
                    self.global_dict['write_summaries_this_step'] = False
                    summary = self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                                         global_step=self.global_dict['counter'], summaries=True)
                    self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                else:
                    self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                               global_step=self.global_dict['counter'], summaries=False)
                self.reset_minibatch()

            self.anneal_epsilon()

    def minibatch_accumulate(self, state, action, reward, state_tp1, terminal, map_data):
        self.minibatch_vars['states'].append(state[0])
        self.minibatch_vars['actions'].append(action)
        self.minibatch_vars['rewards'].append(reward)
        self.minibatch_vars['state_tp1s'].append(state_tp1[0])
        self.minibatch_vars['terminals'].append(terminal)
        self.minibatch_vars['map_data'].append(map_data[0])

    def reset_minibatch(self):
        self.minibatch_vars['states'] = []
        self.minibatch_vars['actions'] = []
        self.minibatch_vars['rewards'] = []
        self.minibatch_vars['state_tp1s'] = []
        self.minibatch_vars['terminals'] = []
        self.minibatch_vars['map_data'] = []

    def get_minibatch_vars(self):
        return [self.minibatch_vars['states'],
                self.minibatch_vars['actions'],
                self.minibatch_vars['rewards'],
                self.minibatch_vars['state_tp1s'],
                self.minibatch_vars['terminals'],
                self.minibatch_vars['map_data']]


class MANIPSA3CWorker(MANIPSBaseThreadLearner):

    def update(self, state, action, reward, next_state, terminal, map_data, map_data_2):

        if self.history_length > 1:
            self.frame_buffer.add_state(state)
            self.map_buffer.add_state(map_data)
            self.map_buffer_2.add_state(map_data_2)

        if self.reward_clip_vals is not None:
            reward = np.clip(reward, *self.reward_clip_vals)

        if not self.testing:
            if self.history_length > 1:
                self.minibatch_accumulate(self.frame_buffer.get_buffer(), action,
                                          reward, self.frame_buffer.get_buffer_add_state(next_state), terminal,
                                          self.map_buffer.get_buffer())
            else:
                self.minibatch_accumulate([state], action, reward, [next_state], terminal, [map_data])

        self.step_count += 1
        self.global_dict['counter'] += 1

        if not self.testing:
            if self.step_count % self.async_update_step == 0 or terminal:
                summaries = self.global_dict['write_summaries_this_step']
                self.agent.anneal_learning_rate(self.global_dict['counter'])
                if summaries:
                    self.global_dict['write_summaries_this_step'] = False
                    summary = self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                                         global_step=self.global_dict['counter'], summaries=True)
                    self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                else:
                    self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                               global_step=self.global_dict['counter'], summaries=False)
                self.reset_minibatch()

            self.anneal_epsilon()

    def minibatch_accumulate(self, state, action, reward, state_tp1, terminal, map_data):
        self.minibatch_vars['states'].append(state[0])
        self.minibatch_vars['actions'].append(action)
        self.minibatch_vars['rewards'].append(reward)
        self.minibatch_vars['state_tp1s'].append(state_tp1[0])
        self.minibatch_vars['terminals'].append(terminal)
        self.minibatch_vars['map_data'].append(map_data[0])

    def reset_minibatch(self):
        self.minibatch_vars['states'] = []
        self.minibatch_vars['actions'] = []
        self.minibatch_vars['rewards'] = []
        self.minibatch_vars['state_tp1s'] = []
        self.minibatch_vars['terminals'] = []
        self.minibatch_vars['map_data'] = []

    def get_minibatch_vars(self):
        return [self.minibatch_vars['states'],
                self.minibatch_vars['actions'],
                self.minibatch_vars['rewards'],
                self.minibatch_vars['state_tp1s'],
                self.minibatch_vars['terminals'],
                self.minibatch_vars['map_data']]


class A3CLSTMWorker(A3CWorker):
    def __init__(self, agent, name, environment, network, global_dict, async_update_steps=5, reward_clip_vals=[-1, 1],
                 using_e_greedy=True, epsilon_annealing_start=1, epsilon_annealing_choices=[0.1, 0.01, 0.5],
                 epsilon_annealing_probabilities=[0.4, 0.3, 0.3],
                 epsilon_annealing_steps=1000000, global_epsilon_annealing=True, report_frequency=1):
        super().__init__(agent, name, environment=environment, network=network, global_dict=global_dict,
                         async_update_steps=async_update_steps, reward_clip_vals=reward_clip_vals,
                         using_e_greedy=using_e_greedy, epsilon_annealing_start=epsilon_annealing_start,
                         epsilon_annealing_choices=epsilon_annealing_choices,
                         epsilon_annealing_probabilities=epsilon_annealing_probabilities,
                         epsilon_annealing_steps=epsilon_annealing_steps,
                         global_epsilon_annealing=global_epsilon_annealing,
                         report_frequency=report_frequency)
        self.config = self.network.get_config()
        self.lstm_state = deepcopy(self.config.get_lstm_state())

    def reset(self):
        super().reset()
        self.lstm_state = deepcopy(self.config.get_lstm_state())

    def update(self, state, action, reward, next_state, terminal):

        if self.history_length > 1:
            self.frame_buffer.add_state(state)

        if self.reward_clip_vals is not None:
            reward = np.clip(reward, *self.reward_clip_vals)

        if not self.testing:
            if self.history_length > 1:
                self.minibatch_accumulate(self.frame_buffer.get_buffer(), action,
                                          reward, self.frame_buffer.get_buffer_add_state(next_state), terminal)
            else:
                self.minibatch_accumulate([state], action, reward, [next_state], terminal)

        self.step_count += 1
        self.global_dict['counter'] += 1

        if not self.testing:
            if self.step_count % self.async_update_step == 0 or terminal:
                summaries = self.global_dict['write_summaries_this_step']
                self.agent.anneal_learning_rate(self.global_dict['counter'])
                if summaries:
                    self.global_dict['write_summaries_this_step'] = False
                    summary = self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                                         global_step=self.global_dict['counter'], summaries=True,
                                                         other=self.lstm_state)
                    self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
                else:
                    self.network.train_network(*self.get_minibatch_vars(), self.agent.current_learning_rate,
                                               global_step=self.global_dict['counter'], summaries=False,
                                               other=self.lstm_state)
                self.reset_minibatch()

                self.lstm_state = deepcopy(self.config.get_lstm_state())

            self.anneal_epsilon()

    # def update(self, state, action, reward, next_state, terminal):
    #
    #     episode_step = self.environment.get_current_steps()
    #
    #     if self.history_length > 1:
    #         self.frame_buffer.add_state_to_buffer(state)
    #
    #     if self.reward_clip_vals is not None:
    #         reward = np.clip(reward, *self.reward_clip_vals)
    #
    #     if episode_step >= 200 and terminal:
    #         print("Good")
    #         terminal = False
    #     elif episode_step < 200 and terminal:
    #         reward=-1
    #
    #     if self.history_length > 1:
    #         self.minibatch_accumulate(self.frame_buffer.get_buffer(), action,
    #                               reward, self.frame_buffer.get_buffer_with(next_state), terminal)
    #     else:
    #         self.minibatch_accumulate([state], action, reward, [next_state], terminal)
    #
    #     self.step_count += 1
    #     self.global_dict['counter'] += 1
    #
    #     if self.step_count % self.async_update_step == 0 or terminal:
    #         if len(self.minibatch_vars['states']) == 30:
    #             summaries = self.global_dict['write_summaries_this_step']
    #             if summaries:
    #                 self.global_dict['write_summaries_this_step'] = False
    #                 summary = self.network.train_network(*self.get_minibatch_vars(),
    #                                                      global_step=self.global_dict['counter'], summaries=True,
    #                                                      other=self.lstm_state)
    #                 self.global_dict['summary_writer'].add_summary(summary, global_step=self.global_dict['counter'])
    #             else:
    #                 self.network.train_network(*self.get_minibatch_vars(),
    #                                            global_step=self.global_dict['counter'], summaries=False,
    #                                            other=self.lstm_state)
    #             self.reset_minibatch()
    #
    #             self.lstm_state = deepcopy(self.config.get_lstm_state())
    #
    #     self.anneal_epsilon()
