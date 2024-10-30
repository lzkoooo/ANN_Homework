# -*- coding = utf-8 -*-
# @Time : 2024/10/3 上午8:50
# @Author : 李兆堃
# @File : Network.py
# @Software : PyCharm

import numpy as np
import GradAlgorithm
import matplotlib.pyplot as plt


class Network:
    def __init__(self, X, D, W, lr, grad_algorithm, batch_size, max_epoch=10):
        self.X = X
        self.D = D
        self.W = W
        self.lr = lr
        self.grad_algorithm = grad_algorithm
        self.batch_size = batch_size
        self.iter_num = int(len(self.X) / self.batch_size + 1)
        self.max_epoch = max_epoch
        self.e_data = []
        self.w_data = [self.W]

    def train(self):
        if len(self.X) % self.batch_size == 0:
            iter_num = int(len(self.X) / self.batch_size)
        else:
            iter_num = int(len(self.X) / self.batch_size + 1)
        for epoch in range(self.max_epoch):
            self.W = eval(f"GradAlgorithm." + self.grad_algorithm + f"(self.X, self.D, self.W, self.lr, self.batch_size, iter_num, self.e_data, self.w_data)")
        print(self.e_data)
        print(self.w_data)
        # self.show_log(epoch, sample_index, net, out, e, Delta_W, self.W)
        pass

    def show_log(self, epoch, sample_index, net, out, e, Delta_W, W):
        print(
            f'Epoch: {epoch}, Sample_index: {sample_index}, Net: {net}, Out: {out}, e: {e}\nDelta_W:\n{Delta_W}\nW:\n{W}')


if __name__ == '__main__':
    X = np.array([[[-1], [-2]], [[2], [-3]], [[-1], [3]]])
    D = [1.7, -2.3, -1.2]
    W = np.array([[1.0], [-0.8]])
    lr = 0.01
    grad_algorithm = 'SGD'
    batch_size = 2
    max_epoch = 2000

    net = Network(X, D, W, lr, grad_algorithm, batch_size, max_epoch)
    net.train()

    plt.plot(range(len(net.e_data)), net.e_data)
    plt.show()
    pass
