# -*- coding = utf-8 -*-
# @Time : 2024/10/3 上午8:50
# @Author : 李兆堃
# @File : Network.py
# @Software : PyCharm

import numpy as np


class Network:
    def __init__(self, X, d, W, lr, grad_algorithm, batch_size, max_epoch=10):
        self.X = X
        self.d = d
        self.W = W
        self.lr = lr
        self.grad_algorithm = grad_algorithm
        self.batch_size = batch_size
        self.iter_num = int(len(self.X) / self.batch_size + 1)
        self.max_epoch = max_epoch

    def train(self):
        # 计算样本输出
        dW = np.zeros_like(self.W)
        e_list = []
        for epoch in range(self.max_epoch):
            for iter in range(self.iter_num):
                current_index = iter * self.batch_size  # 当前批次起始索引
                tail_index = -1 if (current_index + self.batch_size) > len(self.X) else current_index + self.batch_size    # 当前批次结束索引   如果剩余样本数量小于批次数量，则将结束索引设置为末尾

                for sample, d in self.X[current_index: tail_index], self.d[current_index: tail_index]:
                    e = self.forward(self.W, sample, d)
                    e_list.append(e)
                    if self.grad_algorithm == 'MBGD':
                        dW += e * sample
                if self.grad_algorithm == 'MBGD':
                    dW = -1 * self.lr * dW / self.batch_size
                pass

    def forward(self, W, X, d):
        net = np.dot(W.T, X)
        out = net  # 无变换函数
        e = d - out
        return e


        # 更新权重
        Delta_W = self.lr * e * X / np.dot(X.T, X)[0][0]
        print(X, Delta_W, self.W)
        self.W += Delta_W
        self.show_log(epoch, sample_index, net, out, e, Delta_W, self.W)

        pass

    def show_log(self, epoch, sample_index, net, out, e, Delta_W, W):
        print(
            f'Epoch: {epoch}, Sample_index: {sample_index}, Net: {net}, Out: {out}, e: {e}\nDelta_W:\n{Delta_W}\nW:\n{W}')




if __name__ == '__main__':
    pass
