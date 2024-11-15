import re
import sys

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QDialog, QPushButton, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QTextEdit, QLabel, QCheckBox
)

from .Train import run_train


class ANNStudy(QDialog):
    def __init__(self):
        super().__init__()
        self.X = None
        self.trans_fun = None
        self.learn_rule = None
        self.weight = None
        self.learning_rate = None

        # 设置主窗口布局
        main_layout = QVBoxLayout()

        self.setWindowTitle("ANN_HomeWork  2.8")
        self.setGeometry(500, 350, 1000, 500)

        # 设置子布局
        layout_data = QHBoxLayout()

        # 左侧区域：按钮
        layout_open_window = QVBoxLayout()
        self.open_sample_button = QPushButton("打开样本选择窗口")
        self.open_sample_button.clicked.connect(self.open_sample_window)
        self.open_trans_func_button = QPushButton("打开变换函数选择窗口")
        self.open_trans_func_button.clicked.connect(self.open_trans_func_window)
        self.open_learn_rule_button = QPushButton("打开学习规则选择窗口")
        self.open_learn_rule_button.clicked.connect(self.open_learn_rule_window)

        layout_open_window.setContentsMargins(130, 50, 130, 0)
        layout_open_window.addWidget(self.open_sample_button, 1)
        layout_open_window.addSpacing(20)
        layout_open_window.addWidget(self.open_trans_func_button, 1)
        layout_open_window.addSpacing(20)
        layout_open_window.addWidget(self.open_learn_rule_button, 1)

        # 右侧区域：输入框
        layout_input = QFormLayout()
        label1 = QLabel("初始权向量：")
        self.input_box1 = QLineEdit()
        label2 = QLabel("学习率：")
        self.input_box2 = QLineEdit()
        label3 = QLabel("最大epoch：")
        self.input_box3 = QLineEdit()
        self.input_box1.setFixedWidth(160)
        self.input_box2.setFixedWidth(160)
        self.input_box3.setFixedWidth(160)

        layout_input.setContentsMargins(130, 70, 130, 0)
        layout_input.setSpacing(30)
        layout_input.addRow(label1, self.input_box1)
        layout_input.addRow(label2, self.input_box2)
        layout_input.addRow(label3, self.input_box3)

        # 底部区域：提交和关闭按钮
        layout_button = QHBoxLayout()
        button1 = QPushButton("提交")
        button2 = QPushButton("关闭")
        button1.clicked.connect(self.open_train_window)
        button2.clicked.connect(self.close)
        layout_button.addWidget(button1)
        layout_button.addWidget(button2)

        # 添加到主布局
        main_layout.addLayout(layout_data, 1)
        main_layout.addLayout(layout_button, 1)
        layout_data.addLayout(layout_open_window, 1)
        layout_data.addLayout(layout_input, 1)

        # 设置窗口主布局
        self.setLayout(main_layout)

    def open_sample_window(self):
        self.sample_window = SampleWindow()
        result = self.sample_window.exec_()
        if result == QDialog.Accepted:
            self.X = self.sample_window.get_data()
            print(self.X)

    def open_trans_func_window(self):
        self.trans_func_window = Trans_Func_Window()
        result = self.trans_func_window.exec_()
        if result == QDialog.Accepted:
            self.trans_fun = self.trans_func_window.get_data()
            print(self.trans_fun)

    def open_learn_rule_window(self):
        self.learn_rule_window = Learn_Rule_Window()
        result = self.learn_rule_window.exec_()
        if result == QDialog.Accepted:
            self.learn_rule = self.learn_rule_window.get_data()
            print(self.learn_rule)

    def open_train_window(self):
        D = []
        X = []
        for i in range(len(self.X)):
            x = [float(val) for val in re.findall(r'-?\d+', self.X[i])]
            if 'D = ' in self.X[i]:
                D.append(x[-1])
                x = np.array(x[:-1]).reshape(-1, 1)
            else:
                D.append(0)
                x = np.array(x).reshape(-1, 1)
            X.append(x)

        trans_func = self.trans_fun[0]
        learn_rule = self.learn_rule[0]

        W = np.array([[float(i)] for i in re.findall(r'-?\d+', self.input_box1.text())])
        lr = float(self.input_box2.text())
        max_epoch = int(self.input_box3.text())

        print(f"初始权向量：{W}")
        print(f"学习率：{lr}")

        self.train_window = Train_Window(X, D, W, lr, trans_func, learn_rule, max_epoch)
        self.train_window.exec_()


class CustomDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setGeometry(650, 500, 700, 200)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def add_OK_Close(self):
        self.button1 = QPushButton("确定")
        self.button1.clicked.connect(self.accept)
        self.layout.addWidget(self.button1)

        self.button2 = QPushButton("关闭")
        self.button2.clicked.connect(self.close)
        self.layout.addWidget(self.button2)

    def addCheckbox(self, text_list):
        for i, text in enumerate(text_list):
            checkbox = QCheckBox(text)
            setattr(self, f"checkbox{i + 1}", checkbox)
            self.layout.addWidget(checkbox)


class SampleWindow(CustomDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("选择样本")
        self.All_X = ['X = (1, -3, 5, 2, 7, 2)^T, D = 3', 'X = (1, -2)^T', 'X = (0, 1)^T', 'X = (2, 3)^T',
                      'X = (1, 1)^T', 'X = (2, 1, -1)^T, D = -1', 'X = (0, -1, -1)^T, D = 1',
                      'X = (2, 0, -1)^T, D = -1', 'X = (1, -2, -1)^T, D = 1']
        self.addCheckbox(self.All_X)
        self.add_OK_Close()

    def get_data(self):
        return [self.All_X[i] for i in range(len(self.All_X)) if getattr(self, f"checkbox{i + 1}").isChecked()]


class Trans_Func_Window(CustomDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("选择变换函数")
        self.All_trans_fun = ['sgn', 'unipo_sigmoid', 'bipo_sigmoid', 'pie_linear', 'Probabilistic', 'None_trans_fun']
        self.addCheckbox(self.All_trans_fun)
        self.add_OK_Close()

    def get_data(self):
        return [self.All_trans_fun[i] for i in range(len(self.All_trans_fun)) if getattr(self, f"checkbox{i + 1}").isChecked()]


class Learn_Rule_Window(CustomDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("选择学习规则")
        self.All_learn_rule = ['Hebbian', 'Perception', 'Delta_rule', 'LMS', 'Correlation', 'Winner_take_all',
                               'Outstar']
        self.addCheckbox(self.All_learn_rule)
        self.add_OK_Close()

    def get_data(self):
        return [self.All_learn_rule[i] for i in range(len(self.All_learn_rule)) if getattr(self, f"checkbox{i + 1}").isChecked()]


class Train_Window(CustomDialog):
    def __init__(self, X, D, W, lr, trans_fun, learn_rule, max_epoch):
        super().__init__()
        self.setWindowTitle("训练结果")
        self.setGeometry(650, 200, 700, 800)
        result = run_train(X, D, W, lr, trans_fun, learn_rule, max_epoch)

        self.text_edit = QTextEdit()
        self.text_edit.setText(result)
        self.layout.addWidget(self.text_edit)
        self.add_OK_Close()