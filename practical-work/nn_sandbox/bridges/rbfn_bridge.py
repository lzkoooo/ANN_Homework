import time

import PyQt5.QtCore

from nn_sandbox.backend.algorithms import RbfnAlgorithm
from . import Bridge, BridgeProperty
from .observer import Observable


class RbfnBridge(Bridge):
    ui_refresh_interval = BridgeProperty(0.0)
    dataset_dict = BridgeProperty({})
    training_dataset = BridgeProperty([])
    testing_dataset = BridgeProperty([])
    current_dataset_name = BridgeProperty('')
    total_epoches = BridgeProperty(10)
    most_correct_rate_checkbox = BridgeProperty(True)
    most_correct_rate = BridgeProperty(0.98)
    acceptable_range = BridgeProperty(0.5)
    initial_learning_rate = BridgeProperty(0.8)
    search_iteration_constant = BridgeProperty(10000)
    cluster_count = BridgeProperty(3)
    test_ratio = BridgeProperty(0.3)
    current_iterations = BridgeProperty(0)
    current_learning_rate = BridgeProperty(0.0)
    best_correct_rate = BridgeProperty(0.0)
    current_correct_rate = BridgeProperty(0.0)
    test_correct_rate = BridgeProperty(0.0)
    has_finished = BridgeProperty(True)
    current_neurons = BridgeProperty([])

    def __init__(self):
        super().__init__()
        self.rbfn_algorithm = None

    @PyQt5.QtCore.pyqtSlot()
    def start_rbfn_algorithm(self):
        self.rbfn_algorithm = ObservableRbfnAlgorithm(
            self,
            self.ui_refresh_interval,
            dataset=self.dataset_dict[self.current_dataset_name],
            total_epoches=self.total_epoches,
            most_correct_rate=self._most_correct_rate,
            acceptable_range=self.acceptable_range,
            initial_learning_rate=self.initial_learning_rate,
            search_iteration_constant=self.search_iteration_constant,
            cluster_count=self.cluster_count,
            test_ratio=self.test_ratio
        )
        self.rbfn_algorithm.start()

    @PyQt5.QtCore.pyqtSlot()
    def stop_rbfn_algorithm(self):
        self.rbfn_algorithm.stop()

    @property
    def _most_correct_rate(self):
        if self.most_correct_rate_checkbox:
            return self.most_correct_rate
        return None


class ObservableRbfnAlgorithm(Observable, RbfnAlgorithm):
    def __init__(self, observer, ui_refresh_interval, **kwargs):
        self.has_initialized = False
        Observable.__init__(self, observer)
        RbfnAlgorithm.__init__(self, **kwargs)
        self.has_initialized = True
        self.ui_refresh_interval = ui_refresh_interval

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'current_iterations' and self.has_initialized:
            self.notify(name, value)
            self.notify('current_neurons', [{
                'mean': neuron.mean.tolist(),
                'standard_deviation': float(neuron.standard_deviation),
                'synaptic_weight': float(neuron.synaptic_weight)
            } for neuron in self._neurons if not neuron.is_threshold])
            self.notify('test_correct_rate', self.test())
        if name in ('best_correct_rate', 'current_correct_rate'):
            self.notify(name, value)
        if name in ('training_dataset', 'testing_dataset') and value is not None:
            self.notify(name, value.tolist())

    def run(self):
        self.notify('has_finished', False)
        self.notify('test_correct_rate', 0)
        super().run()
        self.notify('current_neurons', [{
            'mean': neuron.mean.tolist(),
            'standard_deviation': float(neuron.standard_deviation),
            'synaptic_weight': float(neuron.synaptic_weight)
        } for neuron in self._neurons if not neuron.is_threshold])
        self.notify('test_correct_rate', self.test())
        self.notify('has_finished', True)

    def _iterate(self):
        super()._iterate()
        # the following line keeps the GUI from blocking
        time.sleep(self.ui_refresh_interval)

    @property
    def current_learning_rate(self):
        ret = super().current_learning_rate
        self.notify('current_learning_rate', ret)
        return ret
