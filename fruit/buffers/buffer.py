import numpy as np


class StateBuffer:
    """
    A buffer that is used to keep a list of historical states.

    :param shape: the current shape
    :param history_length: the maximum number of historical states
    :dtype dtype: the type of the state
    """
    def __init__(self, shape, history_length=4, dtype=np.uint8):
        self.history_length = history_length
        self.shape = [1] + shape
        self.state_buffer = np.zeros(shape, dtype=dtype)
        self.dtype = dtype

    def add_state(self, state):
        """
        Add a state into the buffer

        :param state: the current state
        """
        self.state_buffer[0, 0:self.history_length - 1] = self.state_buffer[0, 1:self.history_length]
        self.state_buffer[0, self.history_length - 1] = state

    def get_buffer_add_state(self, state):
        """
        Add a state into the buffer and return the buffer

        :param state: the current state
        :return: the specified buffer
        """
        buffer = np.zeros(self.shape, dtype=self.dtype)
        buffer[0, 0:self.history_length - 1] = self.state_buffer[0, 1:self.history_length]
        buffer[0, self.history_length - 1] = state
        return buffer

    def reset(self):
        """
        Reset the environment
        """
        self.state_buffer = np.zeros(self.shape, dtype=self.dtype)

    def get_buffer(self):
        """
        Get current state buffer

        :return: copy of the current state
        """
        return np.copy(self.state_buffer)

    def set_buffer(self, state_buffer):
        """
        Set a new ``state_buffer``.

        :param state_buffer: the current state
        :return: state buffer
        """
        self.state_buffer = state_buffer
