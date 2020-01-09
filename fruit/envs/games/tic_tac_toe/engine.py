from fruit.envs.games.engine import BaseEngine
import numpy as np
import itertools


class TicTacToe(BaseEngine):
    def __init__(self):
        self.width = 3
        self.height = 3
        self.current_state = [[0 for _ in range(self.height)] for _ in range(self.width)]
        self.enemy_mode = 0  # Random mode
        self.all_states = self.__generate_states()

    def __generate_states(self):
        cbs = [p for p in itertools.product([0, 1, 2], repeat=self.width*self.height)]
        valid_moves = []
        i = 0
        for c in cbs:
            print(i)
            i += 1
            if self.__is_valid(c):
                c = np.asarray(c)
                if c in valid_moves:
                    continue

                c_arr = np.reshape(c, [self.width, self.height])
                c_90 = np.rot90(c_arr)
                c_90_f = c_90.flatten()
                if c_90_f in valid_moves:
                    continue

                c_180 = np.rot90(c_90)
                c_180_f = c_180.flatten()
                if c_180_f in valid_moves:
                    continue

                valid_moves.append(c_arr)
        print(len(valid_moves))
        return valid_moves

    def __is_win(self, c):
        def __check(c):
            if c[0].count(1) == 3 or c[1].count(1) == 3 or c[2].count(1) == 3 \
                or c[0].count(2) == 3 or c[1].count(2) == 3 or c[2].count(2) == 3 \
                    or list(np.diagonal(c)).count(1) == 3 or list(np.diagonal(c)).count(2) == 3:
                return True

        if __check(c):
            return True
        c = c.transpose()
        if __check(c):
            return True

        return False

    def __is_valid(self, c):
        num_1 = c.count(1)
        num_2 = c.count(2)
        if abs(num_1-num_2) > 1:
            return False
        return True

    def get_game_name(self):
        return 'TIC TAC TOE'

    def clone(self):
        return TicTacToe()

    def get_num_of_objectives(self):
        return 1

    def get_num_of_agents(self):
        return 1

    def __convert_state(self):
        return None

    def reset(self):
        for i in range(self.width * self.height):
            self.states[i] = 0
        return self.__convert_state()

    def print(self):
        for i in range(self.height):
            st = '| '
            for j in range(self.width):
                k = i * self.width + j

                if self.states[k] == 0:
                    st += '_ | '
                elif self.states[k] == 1:
                    st += 'O | '
                elif self.states[k] == 2:
                    st += 'X | '
            print(st)

    def step(self, action, is_enemy=False):
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

    def render(self):
        raise NotImplemented

    def get_state(self):
        return self.__convert_state()

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

    def is_terminal(self):
        is_full = True
        for s in self.states:
            if s == 0:
                is_full = False

        if is_full:
            return True

        # Search rows
        rows = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        for r in rows:
            if self.__get_reward(r) != 0:
                return True

        # Search cols
        cols = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
        for c in cols:
            if self.__get_reward(c) != 0:
                return True

        # Search digs
        digs = [[0, 4, 8], [2, 4, 6]]
        for d in digs:
            if self.__get_reward(d) != 0:
                return True

        return False

    def get_state_space(self):
        from fruit.types.priv import Space
        return Space(0, 3**len(self.states), True)

    def get_action_space(self):
        return range(self.width * self.height)

    def get_num_of_actions(self):
        return self.width * self.height


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
