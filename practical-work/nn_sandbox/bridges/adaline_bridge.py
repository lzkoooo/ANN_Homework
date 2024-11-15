import time

import PyQt5.QtCore

from ..backend.algorithms import AdalineAlgorithm
from . import Bridge, BridgeProperty
from .observer import Observable


class AdalineBridge(Bridge):
    ui_refresh_interval = BridgeProperty(0.0)
    dataset_dict = BridgeProperty({})
    training_dataset = BridgeProperty([])
    testing_dataset = BridgeProperty([])
    current_dataset_name = BridgeProperty('')
    total_epoches = BridgeProperty(10)
    most_correct_rate_checkbox = BridgeProperty(True)
    most_correct_rate = BridgeProperty(0.98)
    initial_learning_rate = BridgeProperty(0.8)
    search_iteration_constant = BridgeProperty(10000)
    momentum_weight = BridgeProperty(0.5)
    test_ratio = BridgeProperty(0.3)
    network_shape = BridgeProperty([5, 5])
    current_iterations = BridgeProperty(0)
    current_learning_rate = BridgeProperty(0.0)
    best_correct_rate = BridgeProperty(0.0)
    current_correct_rate = BridgeProperty(0.0)
    test_correct_rate = BridgeProperty(0.0)
    has_finished = BridgeProperty(True)
    apply_topic = BridgeProperty(0)

    def __init__(self):
        super().__init__()
        self.adaline_algorithm = None

    @PyQt5.QtCore.pyqtSlot()
    def start_adaline_algorithm(self):
        self.adaline_algorithm = ObservableAdalineAlgorithm(
            self,
            self.ui_refresh_interval,
            dataset=self.dataset_dict[self.current_dataset_name],
            total_epoches=self.total_epoches,
            most_correct_rate=self._most_correct_rate,
            initial_learning_rate=self.initial_learning_rate,
            test_ratio=self.test_ratio,
        )
        self.adaline_algorithm.start()

    @PyQt5.QtCore.pyqtSlot()
    def stop_adaline_algorithm(self):
        self.adaline_algorithm.stop()

    @PyQt5.QtCore.pyqtSlot()
    def apply_adaline_algorithm(self):
        self.adaline_algorithm.apply(self.apply_topic)

    @property
    def _most_correct_rate(self):
        if self.most_correct_rate_checkbox:
            return self.most_correct_rate
        return None


class ObservableAdalineAlgorithm(Observable, AdalineAlgorithm):
    def __init__(self, observer, ui_refresh_interval, **kwargs):
        Observable.__init__(self, observer)
        AdalineAlgorithm.__init__(self, **kwargs)
        self.ui_refresh_interval = ui_refresh_interval

    def __setattr__(self, name, value):  # 自动拦截赋值
        super().__setattr__(name, value)
        if name == 'current_iterations':
            self.notify(name, value)
            self.notify('test_correct_rate', self.test())
        elif name in ('best_correct_rate', 'current_correct_rate'):
            self.notify(name, value)
        elif name in ('training_dataset', 'testing_dataset') and value is not None:
            self.notify(name, value.tolist())

    def run(self):
        self.notify('has_finished', False)
        self.notify('test_correct_rate', 0)
        super().run()
        self.notify('test_correct_rate', self.test())
        self.notify('has_finished', True)
        self.notify('apply_result', self.apply_result)

    def _iterate(self):
        super()._iterate()
        # the following line keeps the GUI from blocking
        time.sleep(self.ui_refresh_interval)

    @property
    def current_learning_rate(self):
        ret = super().current_learning_rate
        self.notify('current_learning_rate', ret)
        return ret
