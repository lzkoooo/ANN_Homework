# -*- coding = utf-8 -*-
# @Time : 2024/10/2 下午6:36
# @Author : 李兆堃
# @File : BP.py
# @Software : PyCharm
import numpy as np


class BP:
    def __init__(self, X, Y, node_list, lr, trans_fun, max_epoch):
        self.X = X
        self.Y = Y
        self.W = []
        prev_layer = [self.X.shape[0], node_list[0]]
        # 隐藏层和输出层共两层
        for layer_index in range(len(node_list)):     # 层数
            layer_W = [2 * np.random.rand(prev_layer[layer_index], 1) - 1 for _ in range(node_list[layer_index])]
            # print(layer_W)
            self.W.append(layer_W)
        self.lr = lr
        self.trans_fun = trans_fun
        self.node_list = node_list
        self.max_epoch = max_epoch
        pass

    def forword(self):
        net_Y = []
        for node_index in range(len(self.node_list[0])):
            net_Y.append(np.dot(self.W[0][node_index].T, self.X))
        net_Y = np.array(net_Y)
        Y = self.trans(net_Y)

        net_O = []
        for node_index in range(len(self.node_list[1])):
            net_O.append(np.dot(self.W[1][node_index].T, Y))
        O = self.trans(net_O)
        
        pass

    def backword(self):
        pass

    def train(self):
        pass

    def trans(self, X):
        F = []
        for x in X:
            f = 0.0
            if self.trans_fun == 'sigmoid':
                f = 1 / (1 + np.exp(-x))
            if self.trans_fun == 'uni_sigmoid':
                f = (1 - np.exp(-x)) / (1 + np.exp(-x))
            F.append(f)
        F = np.array(F)
        return F

if __name__ == '__main__':
    X = np.array([[-1], [1]])
    Y = np.array([[0.95], [0.05]])
    node_each_layer = [3, 5]
    net = BP(X, Y, node_each_layer, 0.1, 'sigmoid', 10000)
    # for layer in range(len(net.W)):
    #     for i in range(len(net.W[layer])):
    #         print(f"layer{layer + 1}_V{i}: {net.W[layer][i]}")
    pass
