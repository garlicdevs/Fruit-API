from fruit.envs.games.engine import BaseEngine
import numpy as np
import itertools


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
        c = c.transpose()
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

    def step(self, actions):
        if self.states[action] == 0:

            if is_enemy:
                self.states[action] = 2
            else:
                self.states[action] = 1

            # Search rows
            rows = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
            for r in rows:
                s = self.__get_reward(r)
                if s != 0:
                    return s

            # Search cols
            cols = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
            for c in cols:
                s = self.__get_reward(c)
                if s != 0:
                    return s

            # Search digs
            digs = [[0, 4, 8], [2, 4, 6]]
            for d in digs:
                s = self.__get_reward(d)
                if s != 0:
                    return s

            if self.enemy_mode == 0: # Random mode
                rs = []
                for i in range(len(self.states)):
                    if self.states[i] == 0:
                        rs.append(i)
                if len(rs) > 0:
                    rn = np.random.randint(0, len(rs))
                    self.states[rs[rn]] = 2

            # Search rows
            for r in rows:
                s = self.__get_reward(r)
                if s != 0:
                    return s

            # Search cols
            for c in cols:
                s = self.__get_reward(c)
                if s != 0:
                    return s

            # Search digs
            for d in digs:
                s = self.__get_reward(d)
                if s != 0:
                    return s

            return 0
        else:
            return -1

    def __get_reward(self, r):
        # Win
        if self.states[r[0]] == 1 and self.states[r[1]] == 1 and self.states[r[2]] == 1:
            return 1
        # Lose
        elif self.states[r[0]] == 2 and self.states[r[1]] == 2 and self.states[r[2]] == 2:
            return -1
        # Not win or lose
        else:
            return 0

    def get_state_space(self):
        from fruit.types.priv import Space
        return Space(0, 3**(self.size*self.size), True)

    def get_action_space(self):
        return range(self.size * self.size)

    def get_num_of_actions(self):
        return self.size * self.size


if __name__ == '__main__':
    game = TicTacToe()

    state = game.reset()
    print('Reset state', state)
    num_actions = game.get_num_of_actions()
    print('Num of Actions', num_actions)

    for i in range(1000):
        rand_action = np.random.randint(0, num_actions)
        print('Action', rand_action)
        reward = game.step(rand_action)
        print('Reward', reward)
        state = game.get_state()
        print('State', state)
        terminal = game.is_terminal()
        game.print()
        print()
        if terminal:
            break
