import collections

import numpy as np

from . import PredictiveAlgorithm
from ..neurons import Perceptron
from ..utils import sigmoid, uni_sigmoid


class BpAlgorithm(PredictiveAlgorithm):
    """ Backpropagation prototype. """

    def __init__(self, dataset, total_epoches=10, most_correct_rate=None,
                 initial_learning_rate=0.8, search_iteration_constant=10000,
                 momentum_weight=0.5, test_ratio=0.3, network_shape=None, activation_function_name='sigmoid'):

        super().__init__(dataset, total_epoches, most_correct_rate,
                         initial_learning_rate, search_iteration_constant,
                         test_ratio)
        self._momentum_weight = momentum_weight

        # the default network shape is (5 * 5)
        self.network_shape = network_shape if network_shape else (5, 5)

        # for momentum
        self._synaptic_weight_diff = collections.defaultdict(lambda: 0)     # 字典中没这个值就默认0

        self.activation_function_name = activation_function_name
        if self.activation_function_name == 'sigmoid':
            self.activation_function = sigmoid
        else:
            self.activation_function = uni_sigmoid

    def _iterate(self):
        result = self._feed_forward(self.current_data[:-1])
        deltas = self._pass_backward(self._normalize(self.current_data[-1]),
                                     result)
        self._adjust_synaptic_weights(deltas)

    def _initialize_neurons(self):
        """ Build the neuron network with single neuron as output layer. """
        self._neurons = tuple((Perceptron(self.activation_function),) * size for size in list(self.network_shape) + [1])   # [5, 5, 1]
        # ( (Perceptron(sigmoid) * 5),  (Perceptron(sigmoid) * 5), (Perceptron(sigmoid) * 1) )

    def _feed_forward(self, data):
        results = [None]
        for idx, layer in enumerate(self._neurons):
            if idx == 0:
                results = get_layer_results(layer, data)    # 这一层的net
                continue
            results = get_layer_results(layer, results)     # 用上一层的data生成这一层的net
        return results[0]

    def _pass_backward(self, expect, result):
        """ Calculate the delta for each neuron. """
        deltas = {}     # 用字典，每个感知器对应每个权值

        if self.activation_function_name == 'sigmoid':
            deltas[self._neurons[-1][0]] = ((expect - result) * result * (1 - result))      # -1为取输出层，后面为输出层误差信号
        else:
            deltas[self._neurons[-1][0]] = ((expect - result) * (1 - result * result)) / 2


        for layer_idx, layer in reversed(tuple(enumerate(self._neurons[:-1]))):
            for neuron_idx, neuron in enumerate(layer):
                if self.activation_function_name == 'sigmoid':
                    deltas[neuron] = (
                        # sum of (delta) * (synaptic weight) for each neuron in next layer
                        sum(deltas[next_layer_units] * next_layer_units.synaptic_weight[neuron_idx]
                            for next_layer_units in self._neurons[layer_idx + 1])   # 每一层都按照 (d-o) * o * (1-o) 计算误差信号
                        * neuron.result
                        * (1 - neuron.result)
                    )
                else:
                    deltas[neuron] = (
                        # sum of (delta) * (synaptic weight) for each neuron in next layer
                        sum(deltas[next_layer_units] * next_layer_units.synaptic_weight[neuron_idx]
                            for next_layer_units in self._neurons[layer_idx + 1])   # 每一层都按照 (d-o) * (1-o_2) / 2 计算误差信号
                        * (1 - neuron.result * neuron.result)
                        / 2
                    )
        return deltas

    def _adjust_synaptic_weights(self, deltas):
        for neuron in deltas:
            self._synaptic_weight_diff[neuron] = (
                self._synaptic_weight_diff[neuron] * self._momentum_weight
                + self.current_learning_rate * deltas[neuron] * neuron.data
            )   # dW = 上一个dW * 动量 + lr * bp的dW * x
            neuron.synaptic_weight += self._synaptic_weight_diff[neuron]

    def _correct_rate(self, dataset):
        if not self._neurons:
            return 0
        correct_count = 0
        for data in dataset:
            self._feed_forward(data[:-1])
            expect = self._normalize(data[-1])
            interval = 1 / (2 * len(self.group_types))
            if expect - interval < self._neurons[-1][0].result < expect + interval:
                correct_count += 1
        if correct_count == 0:
            return 0
        return correct_count / len(dataset)

    def _normalize(self, value):
        """ Normalize expected output. """
        return (2 * (value - np.amin(self.group_types)) + 1) / (2 * len(self.group_types))


def get_layer_results(layer, data):
    for neuron in layer:    # 给这一层的Perceptron(sigmoid)装载数据
        neuron.data = data
    return np.fromiter((neuron.result for neuron in layer), dtype=float)    # 返回这一层的网络输出net
