import time

import PyQt5.QtCore
from . import Bridge, BridgeProperty
from ..backend.algorithms import GradientDescent as GradDialog


class GradBridge(Bridge):

    def __init__(self):
        super().__init__()

    @PyQt5.QtCore.pyqtSlot()
    def start_gradient_descent(self):
        dialog = GradDialog()
        dialog.exec_()
        pass

    @PyQt5.QtCore.pyqtSlot()
    def stop_start_gradient_descent(self):
        pass
