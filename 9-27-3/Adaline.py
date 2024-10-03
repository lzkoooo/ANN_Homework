# -*- coding = utf-8 -*-
# @Time : 2024/9/26 上午8:50
# @Author : 李兆堃
# @File : Network.py
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
            for sample_index in range(len(self.X)):
                X = self.X[sample_index]
                d = self.d[sample_index]
                W_T = self.W.T

                # 计算net
                net = np.dot(W_T, X)[0][0]
                # 计算out
                out = self.tf.get_out(net)
                # 计算误差
                e = d - out
                # 更新权重
                Delta_W = self.eta * e * X / np.dot(X.T, X)[0][0]
                print(X, Delta_W, self.W)
                self.W += Delta_W

                if np.all(Delta_W) == 0:
                    break_count += 1

                self.show_log(epoch, sample_index, net, out, e, Delta_W, self.W)
            # 若所有的样本均无误差则停止训练
            if break_count == len(self.X):
                break
            pass

    def show_log(self, epoch, sample_index, net, out, e, Delta_W, W):
        print(f'Epoch: {epoch}, Sample_index: {sample_index}, Net: {net}, Out: {out}, e: {e}\nDelta_W:\n{Delta_W}\nW:\n{W}')

    def predict(self, test_X):
        pred = []
        for sample_index in range(len(test_X)):
            X = test_X[sample_index]
            W_T = self.W.T
            # 计算net
            net = np.dot(W_T, X)[0][0]
            # 计算out
            out = self.tf.get_out(net)
            if out == -1:
                pred.append('中国')
            elif out == 1:
                pred.append('俄罗斯')
        return pred
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
