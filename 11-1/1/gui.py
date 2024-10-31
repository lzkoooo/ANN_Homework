# -*- coding = utf-8 -*-
# @Time : 2024/10/28 上午11:30
# @Author : 李兆堃
# @File : main.py
# @Software : PyCharm

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QWidget, QPushButton, QLineEdit
from dhnnNetwork import HopfieldNetwork


class HopfieldVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.network = HopfieldNetwork(5)
        x1 = np.array([-1, -1, 1, 1, 1]).reshape(-1, 1)
        x2 = np.array([-1, -1, 1, 1, -1]).reshape(-1, 1)
        x3 = np.array([-1, -1, -1, 1, -1]).reshape(-1, 1)
        x4 = np.array([-1, 1, -1, 1, -1]).reshape(-1, 1)
        x5 = np.array([1, -1, 1, 1, -1]).reshape(-1, 1)
        self.patterns = np.array([x1, x2, x3, x4, x5])
        self.attractors_index = []
        self.epoch = 1
        self.initUI()

    def initUI(self):
        self.setWindowTitle('人工神经网络11-1-1')
        self.setGeometry(400, 400, 1200, 1200)

        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)

        # 显示神经元初态和创建吸引子label
        self.state_title_label = QLabel(f"神经元状态: ", self)
        self.layout.addWidget(self.state_title_label)
        self.state_labels = []
        self.attractors_labels = []
        for i, pattern in enumerate(self.patterns):
            self.state_labels.append(QLabel(f"x{i}: " + str(pattern), self))
            self.attractors_labels.append(QLabel("", self))
        for state_label in self.state_labels:
            self.layout.addWidget(state_label)

        # 初始化吸引子
        self.attractors_title_label = QLabel(f"吸引子: ", self)
        self.layout.addWidget(self.attractors_title_label)
        for attractors_label in self.attractors_labels:
            attractors_label.setVisible(False)
            self.layout.addWidget(attractors_label)

        self.epoch_label = QLabel("epoch: 0", self)
        self.epoch_label.setVisible(False)
        self.layout.addWidget(self.epoch_label)

        self.input_epoch_label = QLabel("请输入epoch:", self)
        self.input_epoch = QLineEdit(self)
        self.layout.addWidget(self.input_epoch_label)
        self.layout.addWidget(self.input_epoch)

        self.train_button = QPushButton('训练', self)
        self.train_button.clicked.connect(self.train_network)
        self.layout.addWidget(self.train_button)

        self.setCentralWidget(self.central_widget)

    def update_neuron_label(self, mode, data):
        if mode == 'state':
            for label, data in zip(self.state_labels, data):
                text = "x" + str(label.text()[1]) + ": " + str(data)
                label.setText(text)
        elif mode == 'attractors':
            for attractor_index, energy in data:
                text = "x" + str(attractor_index) + ": " + str(self.patterns[attractor_index]) + "\n能量: " + str(energy)
                self.attractors_labels[attractor_index].setVisible(True)
                self.attractors_labels[attractor_index].setText(text)

    def train_network(self):
        self.input_epoch_label.setVisible(False)
        self.input_epoch.setVisible(False)
        self.train_button.setVisible(False)
        self.epoch_label.setVisible(True)
        # 初始状态不显示
        self.state_title_label.setVisible(False)
        for state_label in self.state_labels:
            state_label.setVisible(False)

        self.network.state = self.patterns

        epoch = int(self.input_epoch.text())
        for i in range(epoch):
            self.network.train()
            self.epoch_label.setText(f"epoch: {i}")
            self.update_neuron_label('state', self.network.state)
            self.search_attractor_energy()

        self.statusBar().showMessage('网络训练完成')

    def search_attractor_energy(self):
        self.attractors_index = self.network.search()
        energys = self.network.energy(self.attractors_index)
        self.update_neuron_label('attractors', zip(self.attractors_index, energys))
        self.statusBar().showMessage('吸引子及能量')


if __name__ == '__main__':
    app = QApplication(sys.argv)

    visualizer = HopfieldVisualizer()
    visualizer.show()

    sys.exit(app.exec_())
