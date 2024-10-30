# -*- coding = utf-8 -*-
# @Time : 2024/10/19 上午11:56
# @Author : 李兆堃
# @File : 上机练习.py
# @Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class SOM:
    def __init__(self, input_dim, grid_size, initial_lr, initial_radius, max_steps):
        self.input_dim = input_dim
        self.grid_size = grid_size
        self.weights = np.random.random((grid_size, grid_size, input_dim))
        self.lr = initial_lr
        self.initial_radius = initial_radius
        self.time_constant = max_steps / np.log(initial_radius)
        self.max_steps = max_steps

    def train(self, inputs):
        for step in range(self.max_steps):
            idx = np.random.randint(len(inputs))
            input_vector = inputs[idx]
            bmu = self.find_bmu(input_vector)
            radius = self.decay_radius(step)
            lr = self.decay_lr(step)
            self.update_weights(bmu, input_vector, lr, radius)

            if step % 200 == 0:
                print(f"训练进度: {step}")  # 每200步输出一次训练进度

    def find_bmu(self, input_vector):
        diff = self.weights - input_vector
        distances = np.linalg.norm(diff, axis=-1)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def decay_radius(self, step):
        return self.initial_radius * np.exp(-step / self.time_constant)

    def decay_lr(self, step):
        return self.lr * (1 - step / self.max_steps)

    def update_weights(self, bmu, input_vector, lr, radius):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                distance_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu))
                if distance_to_bmu < radius:
                    influence = np.exp(-(distance_to_bmu**2) / (2 * (radius**2)))
                    self.weights[i, j] += influence * lr * (input_vector - self.weights[i, j])

    def map_inputs(self, inputs):
        mappings = []
        for input_vector in inputs:
            bmu = self.find_bmu(input_vector)
            mappings.append(bmu)
        return mappings


class SOMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.som = SOM(input_dim=4, grid_size=5, initial_lr=0.5, initial_radius=2, max_steps=10000)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('人工神经网络-som上机练习')
        self.setGeometry(100, 100, 600, 400)

        # 创建开始训练按钮
        self.start_button = QPushButton('开始训练', self)
        self.start_button.clicked.connect(self.start_training)

        # 创建用于显示绘图的画布
        self.canvas = FigureCanvas(plt.Figure())

        layout = QVBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_training(self):
        inputs = np.array([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 2, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 1]
        ])
        self.som.train(inputs)
        mappings = self.som.map_inputs(inputs)
        self.plot_mappings(mappings)

    def plot_mappings(self, mappings):
        fig, ax = plt.subplots(figsize=(5, 5))

        # 绘制网格
        ax.set_xticks(np.arange(0, self.som.grid_size, 1))
        ax.set_yticks(np.arange(0, self.som.grid_size, 1))
        ax.grid(True)

        # 映射输入模式到输出平面
        input_labels = ['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5']
        for idx, (x, y) in enumerate(mappings):
            ax.text(x, y, input_labels[idx], ha='center', va='center', color='red', fontsize=12)

        ax.set_xlim(-0.5, self.som.grid_size - 0.5)
        ax.set_ylim(-0.5, self.som.grid_size - 0.5)
        ax.invert_yaxis()  # 确保(0, 0)在左上角
        self.canvas.figure = fig
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication([])
    som_app = SOMApp()
    som_app.show()
    app.exec_()
