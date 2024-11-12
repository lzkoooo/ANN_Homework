import collections

import numpy as np

from . import PredictiveAlgorithm
from ..neurons import Perceptron
from ..utils import linear


class AdalineAlgorithm(PredictiveAlgorithm):
    """ Backpropagation prototype. """

    def __init__(self, dataset, total_epoches=10, most_correct_rate=None,
                 initial_learning_rate=0.8, search_iteration_constant=10000,
                 momentum_weight=0.5, test_ratio=0.3, network_shape=None):
        super().__init__(dataset, total_epoches, most_correct_rate,
                         initial_learning_rate, search_iteration_constant,
                         test_ratio)
        # for momentum
        self._synaptic_weight_diff = collections.defaultdict(lambda: 0)

    def _iterate(self):
        result = self._feed_forward(self.current_data[:-1])
        self._adjust_synaptic_weights(self._normalize(self.current_data[-1]), result)

    def _initialize_neurons(self):
        """ Build the neuron network with single neuron as output layer. """
        self._neuron = Perceptron(linear)

    def _feed_forward(self, data):
        results = [None]
        results = get_layer_results(self._neuron, results)
        return results[0]

    def _adjust_synaptic_weights(self, expect, result):
        self._synaptic_weight_diff = self.current_learning_rate * (expect - result) * self._neuron.data / np.dot(self._neuron.data.T, self._neuron.data)[0][0]
        self._neuron.synaptic_weight += self._synaptic_weight_diff

    def _normalize(self, value):
        """ Normalize expected output. """
        return (2 * (value - np.amin(self.group_types)) + 1) / (2 * len(self.group_types))


def get_layer_results(neuron, data):
    neuron.data = data
    return np.fromiter(neuron.result, dtype=float)
