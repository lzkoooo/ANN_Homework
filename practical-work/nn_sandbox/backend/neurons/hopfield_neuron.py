import numpy as np


class HopfieldNeuron:
    def __init__(self, activation_function):
        self._weight: np.ndarray = None
        self.activation_function = activation_function
        self._state: np.ndarray = None

    @property
    def state(self):
        return self._state

    @property
    def weight(self):
        return self._weight

    @state.setter
    def state(self, value):
        self._state = value

    @weight.setter
    def weight(self, value):
        self._weight = value

    def get_result(self, x_state):
        return self.activation_function(np.dot(self._weight, x_state))
