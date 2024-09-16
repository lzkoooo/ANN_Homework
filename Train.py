# -*- coding = utf-8 -*-
# @Time : 2024/9/12 下午1:31
# @Author : 李兆堃
# @File : Train.py
# @Software : PyCharm

import re

import numpy as np

import Learn_Rule
import Trans_Fun


def select(data):
    for count in range(len(data)):
        print('{}、 {}'.format(count + 1, data[count]))
        # print(type(i))
    select_data = [data[int(i) - 1] for i in re.findall(r'\d+', input("请选择第几个:"))]  # i - 1即第一个x下标为0
    # print(select)
    return select_data


def select_data():
    All_X_ori = [[[1], [-2]], [[0], [1]], [[2], [3]], [[1], [1]], [[2], [1], [-1]], [[0], [-1], [-1]], [[2], [0], [-1]],
                 [[1], [-2], [-1]], [[1], [-3], [5], [2], [7], [2]]]
    All_X = [np.array(x) for x in All_X_ori]
    All_trans_fun = ['sgn', 'unipo_sigmoid', 'bipo_sigmoid', 'pie_linear', 'Probabilistic', 'None_trans_fun']
    All_learn_rule = ['Hebbian', 'Perception', 'Delta_rule', 'LMS', 'Correlation', 'Winner_take_all', 'Outstar']

    print('样本目录：')
    select_X = select(All_X)

    print('变换函数目录：')
    select_trans_fun = select(All_trans_fun)[0]

    print('学习规则目录：')
    select_learn_rule = select(All_learn_rule)[0]

    return select_X, select_trans_fun, select_learn_rule
    pass


def one_train(x: np.ndarray, W: np.ndarray, lr: float, trans_fun: str, learn_rule: str, d):
    W_t = W.reshape(1, -1)
    net = np.dot(W_t, x)[0][0]  # 计算网络输入
    print('网络输入：', net)
    out = getattr(Trans_Fun, trans_fun)(net)  # 计算网络输出
    dw = getattr(Learn_Rule, learn_rule)(x, out, lr, d, trans_fun)  # 计算权重更新
    print('权重更新：', dw)
    W += dw  # 更新权重
    return W, dw
    pass


if __name__ == '__main__':
    W = np.array([[0], [1], [0]])  # 初始权向量
    lr = 1
    d = [-1, 1]

    X, trans_fun, learn_rule = select_data()
    print('选择的数据：\n样本：{}\n变换函数：{}\n学习规则：{}'.format(X, trans_fun, learn_rule))
    epoch = 0
    while True:
        dW = None
        for count in range(len(X)):
            print('第{}轮迭代_第{}个样本: {}'.format(epoch + 1, count + 1, X[count]))
            W, dW = one_train(X[count], W, lr, trans_fun, learn_rule, d[count])
        if np.all(dW == 0):  # 终止条件
            break
        epoch += 1
        if epoch == 100: break  # 训练100次也停止
    pass
