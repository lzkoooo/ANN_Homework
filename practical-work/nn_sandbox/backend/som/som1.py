import numpy as np
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QMessageBox,
    QApplication,
)


class SelfOrganizingMap:
    def __init__(self, num_neurons, input_dim, learning_rate=0.1, epochs=100):
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(num_neurons, input_dim)

    def find_bmu(self, x):
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


class SOMApp1(QDialog):
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
        self.inputField.setPlaceholderText('请输入测试向量 (用逗号分隔，例如 1,1,0,0)')
        layout.addWidget(self.inputField)

        self.predictButton = QPushButton('测试输入')
        self.predictButton.clicked.connect(self.test_input)
        layout.addWidget(self.predictButton)

        self.resultLabel = QLabel('')
        layout.addWidget(self.resultLabel)

        self.setLayout(layout)
        self.resize(800, 800)

    def train_som(self):
        data = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        self.som.train(data)
        weights_text = "\n".join([f"神经元 {i}: {self.som.weights[i]}" for i in range(len(self.som.weights))])
        self.weightsLabel.setText(f'最终权重:\n{weights_text}')

    def test_input(self):
        input_text = self.inputField.text()
        try:
            input_vector = np.array([float(x) for x in input_text.split(',')]).reshape(1, -1)
            bmu_index = self.som.predict(input_vector)
            result = {0: 'C', 1: 'I', 2: 'T'}.get(bmu_index, '未知类别')
            QMessageBox.information(self, "分类结果", f'分类结果是：{result}')
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的向量 (例如 1,0,0,0)")
