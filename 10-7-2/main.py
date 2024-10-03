# -*- coding = utf-8 -*-
# @Time : 2024/10/2 下午6:36
# @Author : 李兆堃
# @File : main.py
# @Software : PyCharm
import numpy as np

from BP import BP

if __name__ == '__main__':
    X = np.array([[-1], [1], [3]])
    Y = np.array([[0.95], [0.05]])
    node_each_layer = []
    lr = None
    trans_fun = None
    max_epoch = None

    net = BP(X, Y, node_each_layer, lr, trans_fun, max_epoch=1000)
    pass