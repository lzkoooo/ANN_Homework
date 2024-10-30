# -*- coding = utf-8 -*-
# @Time : 2024/10/20 下午11:40
# @Author : 李兆堃
# @File : 4-10.py
# @Software : PyCharm
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QPushButton,
                             QMessageBox)


class LVQ:
    def __init__(self):
        self.weights = np.array([
            [[-1], [1], [-1]],
            [[1], [-1], [-1]],
            [[-1], [-1], [1]],
            [[1], [-1], [1]],
            [[1], [1], [-1]],
            [[-1], [-1], [-1]],
            [[-1], [1], [1]]
        ])

    def classify(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=(1, 2))
        nearest_index = np.argmin(distances)
        return nearest_index // 2 + 1


class LVQApp(QWidget):
    def __init__(self):
        super().__init__()
        self.lvq = LVQ()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.label = QLabel('输入向量:')
        layout.addWidget(self.label)

        self.input_line = QLineEdit(self)
        layout.addWidget(self.input_line)

        self.classify_button = QPushButton('分类', self)
        self.classify_button.clicked.connect(self.classify_vector)
        layout.addWidget(self.classify_button)

        self.setLayout(layout)
        self.setWindowTitle('人工神经网络作业-4.10-李兆堃')
        self.setGeometry(100, 100, 300, 150)

    def classify_vector(self):
        input_str = self.input_line.text()
        input_vector = np.array([[float(x)] for x in input_str.split(',')])
        class_label = self.lvq.classify(input_vector)
        QMessageBox.information(self, '分类结果', f'该向量属于类{class_label}。')



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LVQApp()
    ex.show()
    sys.exit(app.exec_())
