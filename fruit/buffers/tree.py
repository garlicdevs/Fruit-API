import numpy as np


class SyncPriorityNode(object):
    def __init__(self, state, action, reward, terminal, priority):
        self.state = state
        self.action = action
        self.reward = reward
        self.terminal = terminal
        self.p = priority


class AsyncPriorityNode(object):
    def __init__(self, state, action, reward, next_state, terminal, priority):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminal = terminal
        self.p = priority


class Node(object):
    def __init__(self, left, right, value):
        self.left = left
        self.right = right
        self.value = value


class SortedTree(object):
    def __init__(self):
        self.root = None
        self.max_node = None
        self.min_node = None

    def append(self, value):
        if self.root is None:
            self.root = Node(None, None, value)
            self.min_node = self.root
            self.max_node = self.root
        else:
            if value <= self.root.value:
                self.__insert_left(self.root, value)
            else:
                self.__insert_right(self.root, value)

    def __insert_left(self, root, value):
        if root.left is None:
            root.left = Node(None, None, value)
            if value < self.min_node.value:
                self.min_node = root.left
        else:
            if value <= root.left.value:
                self.__insert_left(root.left, value)
            else:
                self.__insert_right(root.left, value)

    def __insert_right(self, root, value):
        if root.right is None:
            root.right = Node(None, None, value)
            if value > self.max_node.value:
                self.max_node = root.right
        else:
            if value <= root.right.value:
                self.__insert_left(root.right, value)
            else:
                self.__insert_right(root.right, value)

    def print_info(self):
        self.__print_info(self.root)

    def __print_info(self, node):
        if node.left is None and node.right is None:
            print(node.value)
        else:
            if node.left is not None:
                self.__print_info(node.left)
            print(node.value)
            if node.right is not None:
                self.__print_info(node.right)

    def get_min(self):
        if self.min_node is None:
            return -1
        else:
            return self.min_node.value

    def get_max(self):
        if self.max_node is None:
            return -1
        else:
            return self.max_node.value


class SyncSumTree(object):
    def __init__(self, alpha=0.6, size=2**19, state_history=4, debug=True):
        level = 0
        while size > 1:
            size = size / 2
            level = level + 1
        self.max_size = 2**level
        print("Sync prioritized replay max size: ", self.max_size)
        self.experiences = [SyncPriorityNode(None, 0, 0, False, 0) for _ in range(self.max_size)]
        self.states = [None for _ in range(self.max_size)]
        self.start_index = 0
        self.current_size = 0
        self.num_of_levels = 0
        self.alpha = alpha
        self.debug = debug
        self.max_p = 0
        self.epsilon = 0.0001
        self.state_history = state_history

        self.__init_memory()

        self.mini_batch_states = []
        self.mini_batch_actions = []
        self.mini_batch_rewards = []
        self.mini_batch_terminals = []
        self.mini_batch_next_states = []
        self.mini_batch_experiences = []
        self.mini_batch_weights = []

    def __reset(self):
        self.mini_batch_states = []
        self.mini_batch_actions = []
        self.mini_batch_rewards = []
        self.mini_batch_terminals = []
        self.mini_batch_next_states = []
        self.mini_batch_experiences = []
        self.mini_batch_weights = []

    def __init_memory(self):
        level = 0
        size = self.max_size
        while size > 1:
            size = size/2
            level = level + 1
        self.num_of_levels = level
        self.table = [None] * level
        for i in range(level):
            self.table[i] = [0] * (2**i)

    def append(self, state=None, action=0, reward=0, next_state=None, terminal=False):

        if self.current_size <= 0:
            priority = 1
            if self.max_p < priority:
                self.max_p = priority
        else:
            priority = self.max_p

        insert_index = (self.start_index + self.current_size) % self.max_size

        current_t = self.experiences[insert_index]
        pre_p = current_t.p
        current_t.state = insert_index
        current_t.action = action
        current_t.reward = reward
        current_t.terminal = terminal
        current_t.p = priority

        if self.current_size < self.max_size:
            self.current_size = self.current_size + 1
            if self.current_size % 2 == 0:
                left = self.experiences[self.current_size-2]
                right = self.experiences[self.current_size-1]
                self.__update(self.num_of_levels-1, self.current_size-2, left.p**self.alpha + right.p**self.alpha)
            else:
                self.__update(self.num_of_levels-1, self.current_size-1,
                              self.experiences[self.current_size-1].p ** self.alpha)
        else:
            self.start_index = (self.start_index + 1) % self.max_size
            self.__modify(self.num_of_levels-1, insert_index, pre_p**self.alpha, priority**self.alpha)

        if self.state_history > 1:
            self.states[insert_index] = state[-1]
        else:
            self.states[insert_index] = state

    def __update(self, new_level, old_index, new_value):
        new_index = int(old_index/2)
        if new_level >= 0:
            self.table[new_level][new_index] = new_value
            if new_index % 2 == 1:
                self.__update(new_level-1, new_index-1,
                              self.table[new_level][new_index-1] + self.table[new_level][new_index])
            else:
                self.__update(new_level-1, new_index, self.table[new_level][new_index])

    def __modify(self, new_level, old_index, old_value, new_value):
        new_index = int(old_index/2)
        if new_level >= 0:
            old_value_2 = self.table[new_level][new_index]
            self.table[new_level][new_index] = self.table[new_level][new_index] - old_value + new_value
            self.__modify(new_level-1, new_index, old_value_2, self.table[new_level][new_index])

    def print_info(self):
        print("########## TREE INFO ##########")
        print("Num of levels: ", self.num_of_levels)
        for i in range(self.num_of_levels):
            size = 2**i
            s = "Level " + str(i+1) + ":"
            for j in range(size):
                s = s + " " + str(self.table[i][j])
            print(s)

    def __get_sample(self, level, index, rand):
        if level < self.num_of_levels-1:
            left_child_index = index*2
            right_child_index = index*2 + 1
            if self.table[level+1][right_child_index] == 0:
                return self.__get_sample(level+1, left_child_index, rand)
            else:
                if rand <= self.table[level+1][left_child_index]:
                    return self.__get_sample(level+1, left_child_index, rand)
                else:
                    return self.__get_sample(level+1, right_child_index, rand - self.table[level+1][left_child_index])
        else:
            return index, rand

    def update_mini_batch(self, indices_, td_errors_):
        j = 0
        if self.debug:
            print(indices_)
            print(td_errors_)
        for i in indices_:
            error = np.abs(td_errors_[j]) + self.epsilon
            self.__modify(self.num_of_levels - 1, i,
                          self.experiences[i].p ** self.alpha, error ** self.alpha)
            self.experiences[i].p = error
            j = j+1

    def get_probability(self, index):
        total_p = self.table[0][0]
        index_p = self.experiences[index].p ** self.alpha
        return index_p/total_p

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

    def get_mini_batch(self, batch_size, current_beta=1):
        if batch_size > self.current_size:
            raise ValueError("Batch size could not be greater than experience replay size")
        self.__reset()
        p_total = self.table[0][0]
        start_index = 0
        dist = p_total/batch_size
        last_element_index = (self.start_index + self.current_size - 1) % self.max_size
        prev_last_element_index = (self.start_index + self.current_size - 2) % self.max_size
        max_weight = 0
        probs = []

        for j in range(batch_size):
            times = 0
            while True:
                times += 1
                if j == batch_size - 1:
                    rand_index = prev_last_element_index
                else:
                    rand = np.random.uniform(0., 1.)
                    rand = rand*dist + start_index
                    start_index = start_index + dist
                    tmp_index, new_i = self.__get_sample(0, 0, rand)
                    left_child_index = tmp_index * 2
                    right_child_index = tmp_index * 2 + 1
                    if right_child_index >= self.current_size:
                        rand_index = left_child_index
                    else:
                        if new_i <= self.experiences[left_child_index].p**self.alpha:
                            rand_index = left_child_index
                        else:
                            rand_index = right_child_index

                if self.max_size == self.current_size:
                    d = rand_index - self.start_index
                    if d < 0:
                        d = d + self.max_size
                    if (d - self.state_history + 1) < 0:
                        rand_index = (self.start_index + self.state_history - 1) % self.max_size

                if rand_index == last_element_index:
                    rand_index = prev_last_element_index

                if rand_index in self.mini_batch_experiences and times < 100:
                    continue

                current_epx = self.experiences[rand_index]
                self.mini_batch_states.append(self.get_state(current_epx.state))
                self.mini_batch_actions.append(current_epx.action)
                self.mini_batch_rewards.append(current_epx.reward)
                self.mini_batch_terminals.append(current_epx.terminal)
                self.mini_batch_experiences.append(rand_index)
                prob = self.get_probability(rand_index)
                probs.append(prob)
                weight = ((1/self.max_size)*(1/prob)) ** current_beta
                if max_weight < weight:
                    max_weight = weight
                self.mini_batch_weights.append(weight)
                if current_epx.terminal:
                    self.mini_batch_next_states.append(self.get_state(current_epx.state))
                else:
                    next_rand_index = (rand_index + 1) % self.max_size
                    self.mini_batch_next_states.append(self.get_state(self.experiences[next_rand_index].state))
                break

        return self.mini_batch_states, self.mini_batch_actions, self.mini_batch_rewards, self.mini_batch_next_states, \
               self.mini_batch_terminals, self.mini_batch_experiences, \
               np.divide(self.mini_batch_weights, max_weight), probs, max_weight


if __name__ == '__main__':
    sum_tree = SyncSumTree(alpha=2, size=2**3)
    sum_tree.print_info()
    sum_tree.append()
    sum_tree.print_info()
    sum_tree.append()
    sum_tree.print_info()
    sum_tree.append()
    sum_tree.print_info()
    sum_tree.append()
    sum_tree.print_info()
    sum_tree.append()
    sum_tree.print_info()
    sum_tree.append()
    sum_tree.print_info()
    sum_tree.append()
    sum_tree.print_info()
    sum_tree.append()
    sum_tree.print_info()
    sum_tree.append()
    sum_tree.print_info()
    sum_tree.append()
    sum_tree.print_info()

    _,_,_,_,_, indices,_,_,_ = sum_tree.get_mini_batch(2)

    td_errors = [1, 4]
    print('Index', indices)
    sum_tree.update_mini_batch(indices, td_errors)
    sum_tree.print_info()

    _, _, _, _, _, indices,_,_,_ = sum_tree.get_mini_batch(2)
    print('Index', indices)
