

class Annealer:
    """
    Anneal a value from ``start`` to ``end`` in ``steps``.

    :param start: initial value
    :param end: end value
    :param steps: the number of steps is used to anneal a value from ``start`` to ``end``
    """
    def __init__(self, start, end, steps):
        self.start_val = start
        self.end_val = end
        self.curr_val = start
        self.steps = steps
        if steps == 0 or steps is None:
            self.diff = 0
        else:
            self.diff = (end - start) / steps

    def anneal(self, steps=1):
        """
        Anneal the current value by the number of steps

        :param steps: steps to anneal
        :return: the current value
        """
        self.curr_val = self.start_val + self.diff * steps
        if self.start_val > self.end_val:
            if self.curr_val < self.end_val:
                self.curr_val = self.end_val
        else:
            if self.curr_val > self.end_val:
                self.curr_val = self.end_val

        return self.curr_val

    def get_current_value(self):
        """
        Get the current value

        :return: the current value
        """
        return self.curr_val


def _unit_test_1():
    annealer = Annealer(0, 10, 5)
    print(annealer.anneal(2))
    annealer.anneal(1)
    print(annealer.get_current_value())
    print(annealer.anneal(2))
    print(annealer.anneal(2))


def _unit_test_2():
    annealer = Annealer(10, 1, 5)
    print(annealer.anneal(1))
    annealer.anneal(1)
    print(annealer.get_current_value())
    print(annealer.anneal(2))
    print(annealer.anneal(2))


if __name__ == '__main__':
    _unit_test_1()

    _unit_test_2()