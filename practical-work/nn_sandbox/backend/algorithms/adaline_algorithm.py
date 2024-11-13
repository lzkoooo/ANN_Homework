import collections
import sys

import numpy as np
from PyQt5.QtGui import QWindow
from PyQt5.QtWidgets import QMainWindow, QLabel, QApplication, QDialog, QVBoxLayout, QHBoxLayout
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from . import PredictiveAlgorithm
from ..neurons import Perceptron
from ..utils import linear


class AdalineAlgorithm(PredictiveAlgorithm):
    """ Backpropagation prototype. """

    def __init__(self, dataset, total_epoches=10, most_correct_rate=None,
                 initial_learning_rate=0.8,test_ratio=0.3):
        super().__init__(dataset, total_epoches, most_correct_rate,
                         initial_learning_rate, 10000, test_ratio)
        # for momentum
        self.topic = None
        self._synaptic_weight_diff = collections.defaultdict(lambda: 0)
        self.apply_result = None

    def _iterate(self):
        result = self._feed_forward(self.current_data[:-1])
        self._adjust_synaptic_weights(self._normalize(self.current_data[-1]), result)

    def _initialize_neurons(self):
        """ Build the neuron network with single neuron as output layer. """
        self._neuron = Perceptron(linear)

    def _feed_forward(self, data):
        result = get_layer_results(self._neuron, data)
        return result

    def _adjust_synaptic_weights(self, expect, result):
        self._synaptic_weight_diff = self.current_learning_rate * (expect - result) * self._neuron.data / np.dot(self._neuron.data.T, self._neuron.data)
        self._neuron.synaptic_weight += self._synaptic_weight_diff

    def _correct_rate(self, dataset):
        if not self._neuron:
            return 0
        correct_count = 0
        for data in dataset:
            self._feed_forward(data[:-1])
            expect = self._normalize(data[-1])
            interval = 1 / (2 * len(self.group_types))
            if expect - interval < self._neuron.result < expect + interval:
                correct_count += 1
        if correct_count == 0:
            return 0
        return correct_count / len(dataset)

    def _normalize(self, value):
        """ Normalize expected output. """
        return (2 * (value - np.amin(self.group_types)) + 1) / (2 * len(self.group_types))

    def apply(self, topic):
        if topic == 3:
            data = np.loadtxt(r"nn_sandbox/assets/other_data/9-27-3-test.txt")
            predict = self._feed_forward(data)
            result = '俄罗斯' if predict >= 0 else '中国'
            apply_result = result
            dialog = PopupDialog(topic, data, apply_result)
            dialog.exec_()
        elif topic == 4:
            data = np.loadtxt(r"nn_sandbox/assets/other_data/9-27-4-test.txt")
            results = []
            for line in data:
                predict = self._feed_forward(line)
                results.append(predict)
            dialog = PopupDialog(topic, data, np.array(results))
            dialog.exec_()
            pass


def get_layer_results(neuron, data):
    neuron.data = data  # 给神经元赋x的值
    return neuron.result      # neuron.result自动计算


class PopupDialog(QDialog):
    def __init__(self, topic, data, apply_result):
        super().__init__()
        self.setWindowTitle("Adaline算法预测结果")
        self.setGeometry(1200, 700, 700, 350)
        if topic == 3:
            layout = QVBoxLayout()
            label_xy = QLabel(f"坐标为：{data}", self)
            layout.addWidget(label_xy)
            label_result = QLabel(f"预测结果为：{apply_result}", self)
            layout.addWidget(label_result)
            self.setLayout(layout)
        elif topic == 4:
            self.figure = plt.Figure(figsize=(5, 4), dpi=100)
            # 创建子图
            self.ax = self.figure.add_subplot(111)
            # 在子图中绘制一些数据
            self.ax.plot([i for i in range(len(apply_result))], apply_result)
            # 将matplotlib图形嵌入到PyQt的界面中
            self.canvas = FigureCanvas(self.figure)
            # 设置布局
            layout = QVBoxLayout(self)
            layout.addWidget(self.canvas)
            self.setLayout(layout)
            # 绘制图形
            self.canvas.draw()
            pass
