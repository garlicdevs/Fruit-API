import numpy as np
import collections as cl


class SyncExperience(object):
    def __init__(self, state, action, reward, terminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.terminal = terminal


class SyncExperienceReplay(object):
    """
        SyncExperienceReplay class is used to store history episodes
        Data in SyncExperienceReplay are then randomly selected to feed the training process
        via helper function get_mini_batch()
        This class is NOT thread safe but better than Async version in terms of memory usage
    """
    def __init__(self, size=1000000, state_history=4):
        if size <= state_history:
            raise ValueError("Experience replay is too small !!!")
        self.experiences = [SyncExperience(None, 0, 0, False) for _ in range(size)]
        self.max_size = size
        self.start_index = 0
        self.current_size = 0
        self.states = [None for _ in range(size)]
        self.state_history = state_history
        self.mini_batch_states = []
        self.mini_batch_actions = []
        self.mini_batch_rewards = []
        self.mini_batch_terminals = []
        self.mini_batch_next_states = []

    def __reset(self):
        self.mini_batch_states = []
        self.mini_batch_actions = []
        self.mini_batch_rewards = []
        self.mini_batch_terminals = []
        self.mini_batch_next_states = []

    def append(self, state, action, reward, next_state, terminal):
        insert_index = (self.start_index + self.current_size) % self.max_size

        current_exp = self.experiences[insert_index]
        current_exp.state = insert_index
        current_exp.action = action
        current_exp.reward = reward
        current_exp.terminal = terminal

        if self.current_size < self.max_size:
            self.current_size = self.current_size + 1
        else:
            self.start_index = (self.start_index + 1) % self.max_size

        self.states[insert_index] = state[-1]

    def get_state(self, index):
        if self.current_size < self.max_size:
            i = index - self.state_history + 1
            if i < 0:
                s = [self.states[0] for _ in range(-i)] + [self.states[j] for j in range(index + 1)]
            else:
                s = self.states[i:index+1]
        else:
            d = index - self.start_index
            if d < 0:
                d = d + self.max_size

            i = d - self.state_history + 1
            if i < 0:
                s = [self.states[self.start_index] for _ in range(-i)] + [self.states[(self.start_index + j) % self.max_size] for j in range(d + 1)]
            else:
                s = [self.states[(index + self.max_size - self.state_history + 1 + j) % self.max_size] for j in range(self.state_history)]
        return s

    def get_mini_batch(self, batch_size):
        if batch_size > self.current_size:
            raise ValueError("Batch size could not be greater than experience replay size")
        self.__reset()
        last_element_index = (self.start_index + self.current_size - 1) % self.max_size
        prev_last_element_index = (self.start_index + self.current_size - 2) % self.max_size
        rands = np.random.randint(0, self.current_size, batch_size)
        rands[-1] = prev_last_element_index
        for i in np.array(rands):
            if self.current_size == self.max_size:
                d = i - self.start_index
                if d < 0:
                    d = d + self.max_size
                if (d - self.state_history + 1) < 0:
                    i = (self.start_index + self.state_history - 1) % self.max_size
            if i == last_element_index:
                i = prev_last_element_index
            current_epx = self.experiences[i]
            self.mini_batch_states.append(self.get_state(current_epx.state))
            self.mini_batch_actions.append(current_epx.action)
            self.mini_batch_rewards.append(current_epx.reward)
            self.mini_batch_terminals.append(current_epx.terminal)
            if current_epx.terminal:
                self.mini_batch_next_states.append(self.get_state(current_epx.state))
            else:
                next_i = (i + 1) % self.max_size
                self.mini_batch_next_states.append(self.get_state(self.experiences[next_i].state))

        return self.mini_batch_states, self.mini_batch_actions, self.mini_batch_rewards, self.mini_batch_next_states, self.mini_batch_terminals


class AsyncExperience(object):
    def __init__(self, state, action, reward, next_state, terminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.terminal = terminal
        self.next_state = next_state


class AsyncExperienceReplay(object):
    """
        AsyncExperienceReplay class is used to store history episodes
        Data in AsyncExperienceReplay are then randomly selected to feed the training process
        via helper function get_mini_batch()
        This class is thread safe and implement in a lazy style
    """
    def __init__(self, size=1000000):
        self.experiences = cl.deque(maxlen=size)

    def append(self, state, action, reward, next_state, terminal):
        self.experiences.append(AsyncExperience(state, action, reward, next_state, terminal))

    def get_mini_batch(self, batch_size):
        state, action, reward, next_state, terminal = [], [], [], [], []
        last_element_index = len(self.experiences)-1
        rands = np.random.randint(0, last_element_index, batch_size)
        rands[-1] = last_element_index
        for i in np.array(rands):
            exp = self.experiences[i]
            state.append(exp.state)
            action.append(exp.action)
            reward.append(exp.reward)
            terminal.append(exp.terminal)
            next_state.append(exp.next_state)

        return state, action, reward, next_state, terminal
