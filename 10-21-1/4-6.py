# -*- coding = utf-8 -*-
# @Time : 2024/10/19 下午8:36
# @Author : 李兆堃
# @File : 4-6.py
# @Software : PyCharm
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QMessageBox


class SelfOrganizingMap:
    def __init__(self, num_neurons, input_dim, learning_rate=0.1, epochs=100):
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.array([item.reshape(-1, 1) for item in np.random.rand(num_neurons, input_dim)])

    def find_bmu(self, x):
        print(self.weights)
        bmu_index = np.argmin(np.linalg.norm(self.weights - x, axis=1))
        return bmu_index

    def update_weights(self, x, bmu):
        self.weights[bmu] += self.learning_rate * (x - self.weights[bmu])

    def train(self, data):
        for epoch in range(self.epochs):
            for x in data:
                bmu = self.find_bmu(x)
                self.update_weights(x, bmu)

    def predict(self, x):
        bmu = self.find_bmu(x)
        return bmu

class SOMApp(QWidget):
    def __init__(self):
        super().__init__()
        self.som = SelfOrganizingMap(num_neurons=3, input_dim=4, learning_rate=0.1, epochs=100)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('人工神经网络作业4.6')

        layout = QVBoxLayout()

        self.trainButton = QPushButton('训练')
        self.trainButton.clicked.connect(self.train_som)
        layout.addWidget(self.trainButton)

        self.weightsLabel = QLabel('展示权重')
        layout.addWidget(self.weightsLabel)

        self.inputField = QLineEdit()
        self.inputField.setPlaceholderText('')
        layout.addWidget(self.inputField)

        self.predictButton = QPushButton('测试输入')
        self.predictButton.clicked.connect(self.test_input)
        layout.addWidget(self.predictButton)

        self.resultLabel = QLabel('')
        layout.addWidget(self.resultLabel)

        self.setLayout(layout)
        self.resize(400, 200)

    def train_som(self):
        data = [[[1], [0], [0], [0]], [[0], [1], [0], [0]], [[0], [0], [1], [0]]]
        self.som.train(data)
        self.weightsLabel.setText(f'最终权重:\n{self.som.weights}')

    def test_input(self):
        input_text = self.inputField.text()
        input_vector = np.array([float(x) for x in input_text.split(',')]).reshape(-1, 1)
        bmu_index = self.som.predict(input_vector)
        if bmu_index == 0:
            result = 'C'
        elif bmu_index == 1:
            result = 'I'
        elif bmu_index == 2:
            result = 'T'
        QMessageBox.information(self, "分类结果", f'分类结果是：{result}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SOMApp()
    window.show()
    sys.exit(app.exec_())
