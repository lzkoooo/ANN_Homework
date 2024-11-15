import numpy as np
from sympy.logic.boolalg import Boolean


class HopfieldNeuron:
    def __init__(self, activation_function):
        self._weight: np.ndarray = None
        self.activation_function = activation_function
        self._pre_state: np.ndarray = None
        self._current_state: np.ndarray = None

    @property
    def state(self):
        return self._current_state

    @property
    def state_change(self):
        return False if self._current_state == self._pre_state else True

    @property
    def weight(self):
        return self._weight

    @state.setter
    def state(self, value):
        self._pre_state = self._current_state
        self._current_state = value

    @weight.setter
    def weight(self, value):
        self._weight = value

    def get_result(self, current_state):
        return self.activation_function(np.dot(self._weight, current_state).all())
