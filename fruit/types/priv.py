import numpy as np
import gym


class Space(object):
    def __init__(self, min_value, max_value, is_discrete):
        self.__min_value = min_value
        self.__max_value = max_value

        if self.is_scaler(min_value) or self.is_scaler(max_value):
            self.__shape = list([1])
        elif isinstance(min_value, (list, np.ndarray, tuple)) and isinstance(max_value, (list, np.ndarray, tuple)):
            shape = np.asarray(min_value).shape
            shape2 = np.asarray(max_value).shape
            if shape != shape2:
                raise ValueError("Shape of min_value and max_value are mismatched !")

            if shape[0] == 0:
                raise ValueError("No value error")

            self.__shape = list(shape)

            if len(min_value) == 1 and shape[0] == 1:
                self.__min_value = min_value[0]
                self.__max_value = max_value[0]
                self.__shape = list([1])
        else:
            raise ValueError("Unsupported type format !")

        self.__discrete = is_discrete

        if is_discrete and (isinstance(min_value, float) or isinstance(max_value, float)):
            raise ValueError('Discrete data cannot be a real number !')

    @staticmethod
    def is_scaler(value):
        return not isinstance(value, (list, tuple, np.ndarray))

    def get_min(self):
        return self.__min_value

    def get_max(self):
        return self.__max_value

    def _get_elements(self, ind):
        ret = list()
        if len(self.__min_value) == ind:
            return [[]]
        for i in range(self.__min_value[ind], self.__max_value[ind]+1):
            temp = self._get_elements(ind+1)
            for j in temp:
                t = list([list([i]) + j])
                ret += t
        return ret

    def get_range(self):
        if self.__discrete:
            if isinstance(self.__min_value, (int, bool)):
                return [i for i in range(self.__min_value, self.__max_value + 1)], True
            else:
                return [self.__min_value, self.__max_value], True
        else:
            return [self.__min_value, self.__max_value], False

    def get_shape(self):
        return self.__shape

    def is_discrete(self):
        return self.__discrete

    @staticmethod
    def convert_openai_space(space):
        from gym.spaces.box import Box
        from gym.spaces.discrete import Discrete
        if isinstance(space, Box):
            return Space(space.low, space.high, False)
        elif isinstance(space, Discrete):
            return Space(0, space.n-1, True)
        else:
            raise ValueError("Does not support other types than Box and Discrete")


def __unit_test():
    env = gym.make("CartPole-v0")

    a = Space.convert_openai_space(env.observation_space)
    print(a.get_min(), a.get_max(), a.is_discrete(), a.get_range(), a.get_shape())

    b = Space.convert_openai_space(env.action_space)
    print(b.get_min(), b.get_max(), b.is_discrete(), b.get_range(), b.get_shape())


if __name__ == '__main__':
    __unit_test()
