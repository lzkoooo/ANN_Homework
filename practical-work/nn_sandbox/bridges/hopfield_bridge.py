import time

import PyQt5.QtCore

from ..backend.algorithms import HopfieldAlgorithm
from . import Bridge, BridgeProperty
from .observer import Observable


class HopfieldBridge(Bridge):
    ui_refresh_interval = BridgeProperty(0.0)
    dataset_dict = BridgeProperty({})
    training_dataset = BridgeProperty([])
    testing_dataset = BridgeProperty([])
    current_dataset_name = BridgeProperty('')
    total_epoches = BridgeProperty(5)
    initial_learning_rate = BridgeProperty(0.8)
    network_shape = BridgeProperty([5, 5])
    current_iterations = BridgeProperty(0)
    current_learning_rate = BridgeProperty(0.0)
    best_correct_rate = BridgeProperty(0.0)
    current_correct_rate = BridgeProperty(0.0)
    test_correct_rate = BridgeProperty(0.0)
    has_finished = BridgeProperty(True)
    hnn_mode = BridgeProperty('')
    train_mode = BridgeProperty('')
    states = BridgeProperty([])
    energys = BridgeProperty([])
    attractors = BridgeProperty([])
    topic = BridgeProperty('')


    def __init__(self):
        super().__init__()
        self.hopfield_algorithm = None

    @PyQt5.QtCore.pyqtSlot()
    def start_hopfield_algorithm(self):
        self.hopfield_algorithm = ObservableHopfieldAlgorithm(
            self,
            self.ui_refresh_interval,
            dataset=self.dataset_dict[self.current_dataset_name],
            total_epoches=self.total_epoches,
            initial_learning_rate=self.initial_learning_rate,
            mode_name=self.hnn_mode,
            train_mode=self.train_mode,
            topic=self.topic
        )
        self.hopfield_algorithm.start()

    @PyQt5.QtCore.pyqtSlot()
    def stop_hopfield_algorithm(self):
        self.hopfield_algorithm.stop()


class ObservableHopfieldAlgorithm(Observable, HopfieldAlgorithm):
    def __init__(self, observer, ui_refresh_interval, **kwargs):
        Observable.__init__(self, observer)
        HopfieldAlgorithm.__init__(self, **kwargs)
        self.ui_refresh_interval = ui_refresh_interval

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'current_iterations':
            self.notify(name, value)
        elif name in ('training_dataset', 'testing_dataset') and value is not None:
            self.notify(name, value.tolist())
        elif name in ('current_states', 'energys', 'attractors'):
            self.notify(name, value)

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
