# -*- coding = utf-8 -*-
# @Time : 2024/10/3 下午6:21
# @Author : 李兆堃
# @File : GradAlgorithm.py
# @Software : PyCharm
import numpy as np


def forward(W, X, d):
    net = np.dot(W.T, X)[0][0]
    out = net  # 无变换函数
    e = out - d
    return e


def BGD(X, D, W, lr, batch_size, iter_num, e_data, w_data):  # 每epoch迭代一次
    e_epoch = np.zeros_like(X[0], dtype=float)
    loss = []
    for iter_index in range(iter_num):
        start_index = iter_index * batch_size
        end_index = (start_index + batch_size) if (start_index + batch_size) <= len(X) else len(X)
        for current_index in range(start_index, end_index):
            x = X[current_index]
            d = D[current_index]
            e = forward(W, x, d)
            loss.append(abs(e))   # 将这轮中的所有的e都保存下来
            e_epoch += e * x  / np.dot(x.T, x)[0][0]  # MSE均方误差的导数
    e_data.append(sum(loss) * 1.0 / len(loss))  # 将这轮中的所有的e都保存下来
    dW = -1 * lr * e_epoch / len(X)
    W += dW
    w_data.append(W)
    return W


def SGD(X, D, W, lr, batch_size, iter_num, e_data, w_data):  # 每epoch迭代一次
    e_epoch = []
    loss = []
    for iter_index in range(iter_num):
        start_index = iter_index * batch_size
        end_index = (start_index + batch_size) if (start_index + batch_size) <= len(X) else len(X)
        for current_index in range(start_index, end_index):
            x = X[current_index]
            d = D[current_index]
            e = forward(W, x, d)
            loss.append(abs(e))  # 将这轮中的所有的e都保存下来
            e_epoch.append(e)  # 将这轮中的所有的e都保存下来
    e_data.append(sum(loss) * 1.0 / len(loss))  # 将这轮中的所有的e都保存下来
    # 随机选取
    select_index = np.random.randint(0, len(X))
    dW = -1 * lr * e_epoch[select_index] * X[select_index] / np.dot(X[select_index].T, X[select_index])[0][0]
    W += dW
    w_data.append(W)
    return W


def MBGD(X, D, W, lr, batch_size, iter_num, e_data, w_data):  # 每batch_size迭代一次
    loss = []
    for iter_index in range(iter_num):

        e_iter = np.zeros_like(X[0], dtype=float)
        start_index = iter_index * batch_size
        end_index = (start_index + batch_size) if (start_index + batch_size) <= len(X) else len(X)
        for current_index in range(start_index, end_index):
            x = X[current_index]
            d = D[current_index]
            e = forward(W, x, d)
            loss.append(abs(e))  # 将这轮中的所有的e都保存下来
            e_iter += e * x / np.dot(x.T, x)[0][0]

        dW = (-1 * 2 * lr / batch_size) * e_iter   # MSE均方误差导数梯度
        W += dW
        w_data.append(W)
    e_data.append(sum(loss) * 1.0 / len(loss))  # 将这轮中的所有的e都保存下来
    return W
