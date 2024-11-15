import numpy as np
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QPushButton,
    QMessageBox,
    QApplication,
    QWidget,
)


class SOFM:
    def __init__(self, map_shape, input_dim, learning_rate=0.1, sigma=None):
        self.map_shape = map_shape
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma or max(map_shape) / 2
        self.weights = np.random.rand(map_shape[1], input_dim, 1)

    def _find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=1)
        bmu_index = np.unravel_index(np.argmin(distances), (self.map_shape[1],))
        return bmu_index

    def _update_weights(self, x, bmu_index):
        self.weights[bmu_index] += self.learning_rate * (x - self.weights[bmu_index])

    def train(self, data, num_epochs):
        for epoch in range(num_epochs):
            for x in data:
                bmu_index = self._find_bmu(x)
                self._update_weights(x, bmu_index)
            self.sigma *= 0.99

    def predict(self, x):
        bmu = self._find_bmu(x)[0]
        return bmu


class SOMApp3(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人工神经网络-10-21-2")
        self.resize(800, 800)

        self.data = np.array([item.reshape(-1, 1) for item in np.array([
            [0.1, 0.3],
            [0.4, 0.2],
            [1.3, 0.4],
            [1.2, 0.2],
            [1.9, 1.9],
            [1.8, 1.9],
            [0.2, 1.8],
            [0.4, 1.8],
            [1.3, 1.7],
            [1.2, 1.9],
        ])])  # 转换为列向量

        self.sofm = SOFM(map_shape=(2, 3), input_dim=2)

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.train_button = QPushButton("开始训练", self)
        self.train_button.clicked.connect(self.train_sofm)
        layout.addWidget(self.train_button)

        self.setLayout(layout)

    def train_sofm(self):
        self.sofm.train(self.data, num_epochs=100)
        self.classify()

    def classify(self):
        result = {}
        for x in self.data:
            bmu_index = self.sofm.predict(x)
            result[f'{x.flatten()}'] = bmu_index
        show = '\n'.join([f'{key}: {value}' for key, value in result.items()])
        QMessageBox.information(self, "训练结果", f'结果是：\n{show}')