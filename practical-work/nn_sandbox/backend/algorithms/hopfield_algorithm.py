import numpy as np
from . import TraningAlgorithm
from ..neurons import HopfieldNeuron
from ..utils import sign, tanh


class HopfieldAlgorithm(TraningAlgorithm):
    """ Backpropagation prototype. """

    def __init__(self, dataset, total_epoches=10, initial_learning_rate=0.8, mode_name='DHNN', train_mode='异步更新',
                 topic='11月1日DHNN第1题'):
        super().__init__(dataset, total_epoches)
        self._dataset = np.array(dataset)
        self.current_states = None

        if topic == '11月1日DHNN第1题':
            self._synaptic_weight_diff = np.loadtxt(r"nn_sandbox/assets/other_data/11-1-1-weight.txt")
            self.network_shape = len(self._dataset)

        self.mode_name = mode_name
        if mode_name == 'DHNN':
            self.activation_function = sign
        elif mode_name == 'CHNN':
            self.activation_function = tanh
        self.train_mode = train_mode
        self.neuron_number = None
        self.update_order = 0
        self._initial_learning_rate = initial_learning_rate
        self.current_iterations = 0

    def run(self):
        self._initialize_neurons()
        for self.current_iterations in range(self._total_epoches * len(self.current_states)):  # 默认单样本为1个iteration
            self._iterate()
            if self._is_stop():
                # 计算吸引子
                break
            # 计算能量

    def _is_stop(self):
        if self._should_stop:
            return True
        for neuron in self._neurons:
            if neuron.state_change:
                return False
        return True

    def _initialize_neurons(self):
        num = 1
        for size in list(self.network_shape):
            num *= size
        self.neuron_number = num
        self._neurons = tuple(HopfieldNeuron(self.activation_function) for _ in range(self.neuron_number))

        for neuron, weidgt in zip(self._neurons, self._synaptic_weight_diff):
            neuron.weight = weidgt

    def _iterate(self, data):
        for data in self._dataset:
            self.current_states =
            for neuron, init_state in zip(self._neurons, data):
                neuron.state = init_state
            self._feed_forward(data)
            self.current_states = self.get_states()

    def _feed_forward(self, data):
        if self.train_mode == '异步更新':
            neuron_idx = self.update_order % self.neuron_number
            neuron = self._neurons[neuron_idx]  # 按顺序进行异步更新
            result = neuron.get_result(data)
            neuron.state = result
            self.update_order += 1
            pass
        elif self.train_mode == '同步更新':
            for neuron in self._neurons:
                result = neuron.get_result(data)
                neuron.state = result

    @property
    def current_learning_rate(self):
        return self._initial_learning_rate / (1 + self.current_iterations / 10000)

    def get_states(self):
        return np.array([neuron.state for neuron in self._neurons])
