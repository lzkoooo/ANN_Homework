import time

import PyQt5.QtCore

from ..backend.algorithms import SomAlgorithm, apply
from . import Bridge, BridgeProperty
from .observer import Observable


class SomBridge(Bridge):
    ui_refresh_interval = BridgeProperty(0.0)
    dataset_dict = BridgeProperty({})
    current_dataset_name = BridgeProperty('')
    total_epoches = BridgeProperty(10)
    initial_learning_rate = BridgeProperty(0.8)
    initial_standard_deviation = BridgeProperty(1.0)
    topology_shape = BridgeProperty([])
    current_iterations = BridgeProperty(0)
    current_learning_rate = BridgeProperty(0.0)
    current_standard_deviation = BridgeProperty(0.0)
    has_finished = BridgeProperty(True)
    current_synaptic_weights = BridgeProperty([])
    apply_topic = BridgeProperty(0)

    def __init__(self):
        super().__init__()
        self.som_algorithm = None

    @PyQt5.QtCore.pyqtSlot()
    def start_som_algorithm(self):
        self.som_algorithm = ObservableSomAlgorithm(
            self,
            self.ui_refresh_interval,
            dataset=self.dataset_dict[self.current_dataset_name],
            total_epoches=self.total_epoches,
            initial_learning_rate=self.initial_learning_rate,
            initial_standard_deviation=self.initial_standard_deviation,
            topology_shape=self.topology_shape
        )
        self.som_algorithm.start()

    @PyQt5.QtCore.pyqtSlot()
    def stop_som_algorithm(self):
        self.som_algorithm.stop()

    @PyQt5.QtCore.pyqtSlot()
    def apply_som_algorithm(self):
        apply(self.apply_topic)

class ObservableSomAlgorithm(Observable, SomAlgorithm):
    def __init__(self, observer, ui_refresh_interval, **kwargs):
        Observable.__init__(self, observer)
        SomAlgorithm.__init__(self, **kwargs)
        self.ui_refresh_interval = ui_refresh_interval

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'current_iterations':
            self.notify(name, value)
            self.notify('current_synaptic_weights',
                        [[neuron.synaptic_weight.tolist()
                          for neuron in row] for row in self._neurons])

    def run(self):
        self.notify('has_finished', False)
        super().run()
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

    @property
    def current_standard_deviation(self):
        ret = super().current_standard_deviation
        self.notify('current_standard_deviation', ret)
        return ret
