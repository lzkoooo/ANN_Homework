import time

import PyQt5.QtCore
from . import Bridge, BridgeProperty
from ..backend.algorithms import BMApp as BmDialog


class BmBridge(Bridge):

    def __init__(self):
        super().__init__()

    @PyQt5.QtCore.pyqtSlot()
    def start_bm(self):
        dialog = BmDialog()
        dialog.exec_()
        pass
