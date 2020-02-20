from fruit.envs.games.engine import BaseEngine
import numpy as np


class TicTacToe(BaseEngine):
    def __init__(self, size=3):
        self.size = size
        self.current_state = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.enemy_mode = 0  # Random mode

    def get_num_of_objectives(self):
        return 1

    def get_num_of_agents(self):
        return 1

    def get_game_name(self):
        return 'TIC TAC TOE'

    def clone(self):
        return TicTacToe(self.size)

    def render(self):
        raise NotImplemented

    def get_state(self):
        return self.__convert_state()

    def __convert_state(self):
        index = 0
        total = 0
        for h in range(self.size):
            for r in range(self.size):
                total += self.current_state[h][r] * (3 ** index)
                index += 1
        return total

    def is_terminal(self):
        is_full = True
        for h in range(self.size):
            for r in range(self.size):
                if self.current_state[h][r] == 0:
                    is_full = False

        if is_full:
            return True

        # Search rows
        for h in range(self.size):
            if self.current_state[h].count(1) == self.size or self.current_state[h].count(2) == self.size:
                return True

        # Search cols
        c = np.asarray(self.current_state)
        c = c.transpose().tolist()
        for h in range(self.size):
            if c[h].count(1) == self.size or c[h].count(2) == self.size:
                return True

        # Search dig 1
        c = np.asarray(self.current_state)
        dig = list(np.diagonal(c))
        if dig.count(1) == self.size or dig.count(2) == self.size:
            return True

        # Search dig 2
        c = np.asarray(self.current_state)
        c = np.rot90(c)
        dig = list(np.diagonal(c))
        if dig.count(1) == self.size or dig.count(2) == self.size:
            return True

        return False

    def reset(self):
        self.current_state = [[0 for _ in range(self.size)] for _ in range(self.size)]
        return self.__convert_state()

    def print(self):
        for h in range(self.size):
            st = '| '
            for r in range(self.size):
                if self.current_state[h][r] == 0:
                    st += '_ | '
                elif self.current_state[h][r] == 1:
                    st += 'O | '
                elif self.current_state[h][r] == 2:
                    st += 'X | '
            print(st)

    def __get_reward(self, arr, is_enemy):
        if not isinstance(arr, list):
            arr = arr.tolist()
        if not is_enemy:
            if arr.count(1) == self.size:
                return 1
            elif arr.count(2) == self.size:
                return -1
            else:
                return 0
        else:
            if arr.count(1) == self.size:
                return -1
            elif arr.count(2) == self.size:
                return 1
            else:
                return 0

    def get_possible_actions(self):
        index = 0
        actions = []
        for h in range(self.size):
            for r in range(self.size):
                if self.current_state[h][r] == 0:
                    actions.append(index)
                index += 1
        return actions

    def step(self, action, is_enemy=False):
        h = int(action / self.size)
        w = action % self.size

        if self.current_state[h][w] == 0:
            if is_enemy:
                self.current_state[h][w] = 2
            else:
                self.current_state[h][w] = 1

            # Search rows
            for h in range(self.size):
                r = self.__get_reward(self.current_state[h], is_enemy)
                if r != 0:
                    return r

            # Search cols
            c = np.asarray(self.current_state)
            c = c.transpose()
            for h in range(self.size):
                r = self.__get_reward(c[h], is_enemy)
                if r != 0:
                    return r

            # Search dig 1
            c = np.asarray(self.current_state)
            dig = list(np.diagonal(c))
            r = self.__get_reward(dig, is_enemy)
            if r != 0:
                return r

            # Search dig 2
            c = np.asarray(self.current_state)
            c = np.rot90(c)
            dig = list(np.diagonal(c))
            r = self.__get_reward(dig, is_enemy)
            if r != 0:
                return r

            return 0
        else:
            return -1

    def get_state_space(self):
        from fruit.types.priv import Space
        return Space(0, 3**(self.size*self.size), True)

    def get_action_space(self):
        return range(self.size * self.size)

    def get_num_of_actions(self):
        return len(self.get_possible_actions())


if __name__ == '__main__':
    game = TicTacToe()

    state = game.reset()
    print('Reset state', state)
    num_actions = game.get_num_of_actions()
    print('Num of Actions', num_actions)

    for i in range(100):
        # Player
        print('------ PLAYER PHASE -----')
        actions = game.get_possible_actions()
        rand_action = np.random.choice(actions)
        print('Action', rand_action)
        reward = game.step(rand_action)
        print('Reward', reward)
        state = game.get_state()
        print('State', state)
        terminal = game.is_terminal()
        game.print()
        if terminal:
            break
        print('------ ----------- -----\n')

        # Enemy
        print('------ ENEMY PHASE -----')
        actions = game.get_possible_actions()
        rand_action = np.random.choice(actions)
        print('Action', rand_action)
        reward = game.step(rand_action, is_enemy=True)
        print('Reward', reward)
        state = game.get_state()
        print('State', state)
        terminal = game.is_terminal()
        game.print()
        print('------ ----------- -----\n')
        if terminal:
            break
