# -*- coding = utf-8 -*-
# @Time : 2024/10/2 19:50
# @Author : 李兆堃
# @File : dhnnNetwork.py
# @Software : PyCharm
import numpy as np


class HopfieldNetwork:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.array(
            [[0, 1, 1, -1, 1], [1, 0, -1, -3, 3], [1, -1, 0, 1, -1], [-1, -3, 1, 0, -3], [1, 3, -1, -3, 0]])  # 初始化权重矩阵
        self.pre_state = np.array([])
        self.current_state = np.array([])

    @property
    def state(self):
        return self.current_state

    @state.setter
    def state(self, value):
        self.current_state = value

    def sgn(self, net):
        for i in range(len(net)):
            if net[i] >= 0:
                net[i] = 1
            else:
                net[i] = -1
        return net

    def train(self):
        self.pre_state = self.current_state
        outs = []

        for pattern in self.current_state:
            net = np.dot(self.weights, pattern)
            out = self.sgn(net)
            outs.append(out)

        self.current_state = np.array(outs)

    def search(self):
        # 寻找吸引子
        attractors_index = []
        for i in range(len(self.current_state)):
            if np.array_equal(self.current_state[i], self.pre_state[i]):
                attractors_index.append(i)
        return attractors_index if len(attractors_index) > 0 else None

    def energy(self, index):
        # 计算能量函数
        es = []
        if len(index) == 0:
            return None

        for i in index:
            e = -0.5 * np.dot(self.state[i].T, np.dot(self.weights, self.state[i]))[0][0]
            es.append(e)

        return es
