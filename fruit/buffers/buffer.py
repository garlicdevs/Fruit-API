import numpy as np


class StateBuffer:
    def __init__(self, shape, history_length=4, dtype=np.uint8):
        self.history_length = history_length
        self.shape = [1] + shape
        self.state_buffer = np.zeros(shape, dtype=dtype)
        self.dtype = dtype

    def add_state(self, state):
        self.state_buffer[0, 0:self.history_length - 1] = self.state_buffer[0, 1:self.history_length]
        self.state_buffer[0, self.history_length - 1] = state

    def get_buffer_add_state(self, state):
        buffer = np.zeros(self.shape, dtype=self.dtype)
        buffer[0, 0:self.history_length - 1] = self.state_buffer[0, 1:self.history_length]
        buffer[0, self.history_length - 1] = state
        return buffer

    def reset(self):
        self.state_buffer = np.zeros(self.shape, dtype=self.dtype)

    def get_buffer(self):
        return np.copy(self.state_buffer)

    def set_buffer(self, state_buffer):
        self.state_buffer = state_buffer
