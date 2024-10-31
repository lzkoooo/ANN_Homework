# -*- coding = utf-8 -*-
# @Time : 2024/10/30 13:21
# @Author : 李兆堃
# @File : bm.py
# @Software : PyCharm
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer


class BoltzmannMachine:
    def __init__(self, num_neurons=3, temperature=1.0, equilibrium_threshold=1e-3):
        self.num_neurons = num_neurons
        self.temperature = temperature
        self.states = np.random.choice([-1, 1], num_neurons)  # 初始化神经元状态
        self.weights = np.random.uniform(-1, 1, (num_neurons, num_neurons))  # 初始化权值
        np.fill_diagonal(self.weights, 0)  # 去掉自连项
        self.previous_energy = self.calculate_energy()  # 记录初始能量
        self.equilibrium_threshold = equilibrium_threshold

    def calculate_energy(self):
        # 计算当前状态下的系统能量
        return -0.5 * np.sum(self.weights * np.outer(self.states, self.states))

    def update_neuron(self, neuron_index):
        # 更新单个神经元的状态
        input_sum = np.dot(self.weights[neuron_index], self.states)
        prob = 1 / (1 + np.exp(-2 * input_sum / self.temperature))
        new_state = 1 if np.random.rand() < prob else -1
        old_state = self.states[neuron_index]

        # 临时更新神经元状态并计算能量变化
        self.states[neuron_index] = new_state
        current_energy = self.calculate_energy()
        energy_diff = current_energy - self.previous_energy
        self.previous_energy = current_energy

        # 如果未达到热平衡则恢复状态
        if abs(energy_diff) >= self.equilibrium_threshold:
            self.states[neuron_index] = old_state  # 恢复原状态

        return energy_diff

    def update_and_check_equilibrium(self):
        # 更新所有神经元并检查每个神经元的能量变化量
        equilibrium_reached = True
        energy_diffs = []

        for neuron_index in range(self.num_neurons):
            energy_diff = self.update_neuron(neuron_index)
            energy_diffs.append(energy_diff)
            if abs(energy_diff) >= self.equilibrium_threshold:
                equilibrium_reached = False  # 任意一个神经元未达到阈值，非热平衡

        return equilibrium_reached, energy_diffs


class BMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Boltzmann Machine Simulation")
        self.bm = BoltzmannMachine()
        self.setGeometry(1000, 500, 1000, 1000)

        # 界面组件
        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.start_simulation)
        self.status_label = QLabel("状态: 未开始")
        self.energy_label = QLabel("能量变化量: 未知")

        layout = QVBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.energy_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 设置定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)

    def start_simulation(self):
        self.bm = BoltzmannMachine()  # 重新初始化BM
        self.update_display(initial=True)
        self.timer.start(100)  # 每500毫秒更新一次

    def update_simulation(self):
        equilibrium_reached, energy_diffs = self.bm.update_and_check_equilibrium()
        self.update_display(energy_diffs)

        # 如果达到热平衡，停止更新
        if equilibrium_reached:
            self.timer.stop()
            self.status_label.setText("状态: 达到热平衡")

    def update_display(self, energy_diffs=None, initial=False):
        # 更新能量变化量的显示
        if initial:
            self.energy_label.setText("能量变化量: 初始化中")
        else:
            energy_diff_text = "\n".join(
                [f"神经元 {i + 1} 能量变化量: {diff:.4f}" for i, diff in enumerate(energy_diffs)])
            self.energy_label.setText(f"能量变化量:\n{energy_diff_text}")
        self.status_label.setText("状态: 运行中")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BMApp()
    window.show()
    sys.exit(app.exec_())
