# -*- coding = utf-8 -*-
# @Time : 2024/10/23 上午9:59
# @Author : 李兆堃
# @File : main.py
# @Software : PyCharm

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QWidget, QPushButton


class HopfieldNetwork:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.array([[0, 1, 1, -1, 1], [1, 0, -1, -3, 3], [1, -1, 0, 1, -1], [-1, -3, 1, 0, -3], [1, 3, -1, -3, 0]])  # 初始化权重矩阵
        self.pre_state = np.array([])
        self.current_state = np.array([])
        print(self.weights)

    def sgn(self, net):
        for i in range(len(net)):
            if net[i] >= 0:
                net[i] = 1
            else:
                net[i] = -1
        return net

    def train(self, patterns):
        self.pre_state = patterns
        outs = []

        for pattern in patterns:
            net = np.dot(self.weights, pattern)
            out = self.sgn(net)
            outs.append(out)

        self.current_state = np.array(outs)

    def search(self):
        # 寻找吸引子
        attractors = []
        for i in range(len(self.current_state)):
            if np.array_equal(self.current_state[i], self.pre_state[i]):
                attractors.append(self.current_state[i])
        return np.array(attractors)

    def energy(self, patterns):
        # 计算能量函数
        es = []
        for pattern in patterns:
            e = -0.5 * np.dot(pattern.T, np.dot(self.weights, pattern))
            es.append(e)

        return np.array(es)


class HopfieldVisualizer(QMainWindow):
    def __init__(self, hopfield_network):
        super().__init__()
        self.network = hopfield_network
        x1 = np.array([-1, -1, 1, 1, 1]).reshape(-1, 1)
        x2 = np.array([-1, -1, 1, 1, -1]).reshape(-1, 1)
        x3 = np.array([-1, -1, -1, 1, -1]).reshape(-1, 1)
        x4 = np.array([-1, 1, -1, 1, -1]).reshape(-1, 1)
        x5 = np.array([1, -1, 1, 1, -1]).reshape(-1, 1)
        self.patterns = np.array([x1, x2, x3, x4, x5])
        self.initUI()

    def initUI(self):
        self.setWindowTitle('人工神经网络10-25')
        self.setGeometry(100, 100, 400, 300)

        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)

        # 显示神经元状态
        self.neuron_label = QLabel(self)
        self.update_neuron_label(self.patterns)
        self.layout.addWidget(self.neuron_label)

        # 显示能量
        self.energy_label = QLabel("能量: 0", self)
        self.layout.addWidget(self.energy_label)

        self.train_button = QPushButton('训练', self)
        self.train_button.clicked.connect(self.train_network)
        self.layout.addWidget(self.train_button)

        self.attractor_energy_button = QPushButton('寻找吸引子并计算能量', self)
        self.attractor_energy_button.clicked.connect(self.search_attractor_energy)
        self.layout.addWidget(self.attractor_energy_button)

        self.setCentralWidget(self.central_widget)

    def update_neuron_label(self, pattern):
        text = "神经元状态: " + str(pattern)
        self.neuron_label.setText(text)

    def train_network(self):
        self.network.train(self.patterns)
        self.statusBar().showMessage('网络训练完成')

    def search_attractor_energy(self):
        attractors = self.network.search()
        energys = self.network.energy(attractors)[0][0][0]
        self.update_neuron_label(attractors)
        self.energy_label.setText(f"能量: {energys:.2f}")
        self.statusBar().showMessage('吸引子及能量')


if __name__ == '__main__':
    app = QApplication(sys.argv)

    network = HopfieldNetwork(5)

    visualizer = HopfieldVisualizer(network)
    visualizer.show()

    sys.exit(app.exec_())
