import sys

import numpy as np
from PyQt5.QtWidgets import QDialog, QApplication

from . import TraningAlgorithm, TSPWindow
from ..neurons import HopfieldNeuron
from ..utils import sign, tanh


def hop_apply():
    cities = [
        [0.1, 0.6], [0.8, 0.3], [0.5, 0.6], [0.4, 0.9],
        [0.7, 0.1], [0.5, 0.5], [0.7, 0.3], [0.8, 0.8],
        [0.2, 0.9], [0.4, 0.1]
    ]
    dialog = TSPWindow(cities)
    dialog.exec_()


class HopfieldAlgorithm(TraningAlgorithm):
    """ Backpropagation prototype. """

    def __init__(self, dataset, total_epoches=10, initial_learning_rate=0.8, mode_name='DHNN', train_mode='异步更新',
                 topic='DHNN_10月25日题目及11月1日第1题'):
        super().__init__(dataset, total_epoches)
        self._dataset = np.array(dataset)
        self.pre_states_matrix = None
        self.new_states_matrix: np.ndarray
        self.new_states_matrix = self._dataset
        self.attractors_idx = []
        self.attractors: np.ndarray
        self.energys: np.ndarray

        if topic == 'DHNN_10月25日题目及11月1日第1题':
            self._synaptic_weight_diff = np.loadtxt(r"nn_sandbox/assets/other_data/11-1-1-weight.txt")
            self.network_shape = (len(self._dataset),)

        self.mode_name = mode_name
        if mode_name == 'DHNN':
            self.activation_function = sign
        elif mode_name == 'CHNN':
            self.activation_function = tanh
        self.train_mode = train_mode
        self.neuron_number = None
        self._initial_learning_rate = initial_learning_rate
        self.current_iterations = 0

    def run(self):
        self._initialize_neurons()
        for self.current_iterations in range(self._total_epoches * len(self._dataset)):  # 默认单样本为1个iteration
            self._iterate()
            self._search_attractors()
            self._calculate_energy()
            if self._is_stop():
                break

    def _is_stop(self):
        if self._should_stop:
            return True
        if np.array_equal(self.pre_states_matrix, self.new_states_matrix):  # 连续5个data都没变
            return True

    def _initialize_neurons(self):
        num = 1
        for size in list(self.network_shape):
            num *= size
        self.neuron_number = num
        self._neurons = tuple(HopfieldNeuron(self.activation_function) for _ in range(self.neuron_number))

        for neuron, weidgt in zip(self._neurons, self._synaptic_weight_diff):
            neuron.weight = weidgt

    def _iterate(self):
        self.pre_states_matrix = self.new_states_matrix
        new_matrix = []
        for idx, data in enumerate(self._dataset):
            self._load_data(data)
            self._feed_forward(data)
            new_states = self._get_states()
            new_matrix.append(new_states)     # 更新原数据
        self.new_states_matrix = self._dataset = np.array(new_matrix)

    def _load_data(self, data):
        for neuron, init_state in zip(self._neurons, data):  # 装载数据
            neuron.state = init_state

    def _feed_forward(self, data):
        if self.train_mode == '异步更新':
            for neuron_idx in range(self.neuron_number):
                neuron = self._neurons[neuron_idx]  # 按顺序进行异步更新
                result = neuron.get_result(data)
                neuron.state = result
            pass
        elif self.train_mode == '同步更新':
            for neuron in self._neurons:
                result = neuron.get_result(data)
                neuron.state = result

    @property
    def current_learning_rate(self):
        return self._initial_learning_rate / (1 + self.current_iterations / 10000)

    def _get_states(self):
        return np.array([neuron.state for neuron in self._neurons])

    def _search_attractors(self):
        attractors = []
        attractors_idx = []
        for idx in range(len(self.pre_states_matrix)):
            if np.array_equal(self.new_states_matrix[idx], self.pre_states_matrix[idx]):
                attractors.append(self.new_states_matrix[idx])
                attractors_idx.append(idx)
        self.attractors = np.array(attractors)
        self.attractors_idx = attractors_idx
        pass

    def _calculate_energy(self):
        if self.attractors is not None:
            es = []
            for idx in self.attractors_idx:
                e = -0.5 * np.dot(self.pre_states_matrix[idx].T, np.dot(self._synaptic_weight_diff, self.pre_states_matrix[idx]))
                es.append(e)
            self.energys = np.array(es)
        pass

