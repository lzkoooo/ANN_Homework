# -*- coding = utf-8 -*-
# @Time : 2024/9/26 上午8:50
# @Author : 李兆堃
# @File : Adaline.py
# @Software : PyCharm
from typing import Literal
from Trans_Fun import Trans_Fun
import numpy as np


class Adaline:
    def __init__(self, X: [], W: np.ndarray, d: [], eta: float, trans_fun: Literal['sgn', 'linear'], max_epoch=10):
        self.X = X
        self.W = W
        self.d = d
        self.eta = eta
        self.tf = Trans_Fun(trans_fun)
        self.max_epoch = max_epoch

    def fit(self):
        for epoch in range(self.max_epoch):
            break_count = 0
            for sample_index in range(len(self.X[0])):
                X = self.X[0]
                d = self.d[sample_index][0]
                W_T = self.W.T[sample_index]

                # 计算net
                net = np.dot(W_T, X)[0]
                # 计算out
                out = self.tf.get_out(net)
                # 计算误差
                e = d - out
                # 更新权重
                Delta_W = self.eta * e * X / np.dot(X.T, X)[0][0]
                # print(X, Delta_W, self.W)
                # for raw in range(len(self.W)):
                self.W[:, sample_index] += Delta_W.flatten()
                print(e)
                if -1e-13 < e < 1e-13:
                    break_count += 1

                # self.show_log(epoch, sample_index, net, out, e, Delta_W, self.W)
            # 若3分之2的误差都在-1e-15~1e-15之间，则停止训练
            if break_count > (len(self.X[0]) / 3 * 2):
                break
            pass

    def show_log(self, epoch, sample_index, net, out, e, Delta_W, W):
        print(f'Epoch: {epoch}, Sample_index: {sample_index}, Net: {net}, Out: {out}, e: {e}\nDelta_W:\n{Delta_W}\nW:\n{W}')

    def predict(self, test_X):
        result = []
        X = test_X[0]
        for sample_index in range(len(X)):
            W_T = self.W.T[sample_index]
            # 计算net
            net = np.dot(W_T, X)[0]
            # 计算out
            out = self.tf.get_out(net)
            result.append(out)
        pred = np.array(result).reshape(-1, 1)
        e = np.array(pred) - X
        return pred, e
        pass

if __name__ == '__main__':
    X = [np.array(i) for i in [[[-1], [1.2], [2.7]]]]
    W = np.array([[-1], [0.5], [1.1]])
    d = [np.array(i) for i in [2.3]]
    eta = 0.6
    max_epoch = 2

    print(f' X: {X}\n W: {W}\n d: {d}\n eta: {eta}\n')
    adaline = Adaline(X, W, d, eta, 'linear', max_epoch)
    adaline.fit()
