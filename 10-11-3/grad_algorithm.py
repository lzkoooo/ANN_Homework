# -*- coding = utf-8 -*-
# @Time : 2024/10/3 下午6:21
# @Author : 李兆堃
# @File : grad_algorithm.py
# @Software : PyCharm
import numpy as np


def BGD(X, d, W, lr, batch_size):       # 每epoch迭代一次
    dW = np.zeros(W.shape)
    for iter in range(int(len(X) / batch_size + 1)):
        current_index = iter * batch_size
        tail_index = current_index + batch_size
        if tail_index > len(X):
            tail_index = -1
        for sample_index in X[current_index: tail_index]:
            net = np.dot(W.T, X)
            out = net  # 无变换函数
            e = d - out
            dW += e * X     # MSE均方误差的导数
    dW = dW / len(X)
    W -= dW * lr
    return W


def SGD(X, d, W, batch_size):       # 每sample迭代一次
    loss = 0.0
    e_epoch = []
    for iter in range(int(len(X) / batch_size + 1)):
        current_index = iter * batch_size
        tail_index = current_index + batch_size
        if tail_index > len(X):
            tail_index = -1
        for sample in X[current_index: tail_index]:
            net = np.dot(W.T, X)
            out = net  # 无变换函数
            e = d - out
            e_epoch.append(e)   # 将这轮中的所有的e都保存下来

    # 随机选取
    select_index = np.random.randint(0, len(e_epoch))
    dW = e_epoch[select_index] * X[select_index]
    return dW


def MBGD(X, d, W, batch_size):      # 每batch_size迭代一次
    loss = 0.0
    for iter in range(int(len(X) / batch_size + 1)):
        e_iter = 0.0
        current_index = iter * batch_size
        tail_index = current_index + batch_size
        if tail_index > len(X):
            tail_index = -1
        for sample in X[current_index: tail_index]:
            net = np.dot(W.T, X)
            out = net  # 无变换函数
            e = d - out
            e_iter += e
        loss = e_iter * e_iter / batch_size  # MSE均方误差
