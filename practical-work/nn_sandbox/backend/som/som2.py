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


class LVQ:
    def __init__(self):
        self.weights = np.array([
            [[-1], [1], [-1]],
            [[1], [-1], [-1]],
            [[-1], [-1], [1]],
            [[1], [-1], [1]],
            [[1], [1], [-1]],
            [[-1], [-1], [-1]],
            [[-1], [1], [1]],
        ])

    def classify(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=(1, 2))
        nearest_index = np.argmin(distances)
        return nearest_index // 2 + 1


class SOMApp2(QDialog):
    def __init__(self):
        super().__init__()
        self.lvq = LVQ()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel('输入向量:')
        layout.addWidget(self.label)

        self.input_line = QLineEdit(self)
        self.input_line.setPlaceholderText('请输入向量 (例如: -1,1,-1)')
        layout.addWidget(self.input_line)

        self.classify_button = QPushButton('分类', self)
        self.classify_button.clicked.connect(self.classify_vector)
        layout.addWidget(self.classify_button)

        self.setLayout(layout)
        self.setWindowTitle('人工神经网络作业-4.10-李兆堃')
        self.resize(800, 800)

    def classify_vector(self):
        input_str = self.input_line.text()
        try:
            input_vector = np.array([[float(x)] for x in input_str.split(',')])
            class_label = self.lvq.classify(input_vector)
            QMessageBox.information(self, '分类结果', f'该向量属于类 {class_label}。')
        except ValueError:
            QMessageBox.warning(self, '输入错误', '请输入有效的向量 (例如: 1,0,-1)。')
