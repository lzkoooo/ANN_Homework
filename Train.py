# -*- coding = utf-8 -*-
# @Time : 2024/9/12 下午1:31
# @Author : 李兆堃
# @File : Train.py
# @Software : PyCharm
import re

import numpy as np


def select(data):
    for count in range(len(data)):
        print('{}、 {}'.format(count + 1, data[count]))
        # print(type(i))
    select_data =[data[int(i) - 1] for i in re.findall(r'\d+', input("请选择第几个:"))]  # i - 1即第一个x下标为0
    # print(select)
    return select_data


def select_data():
    All_X_ori = [[[1], [-2]], [[0], [1]], [[2], [3]], [[1], [1]], [[2], [1], [-1]], [[0], [-1], [-1]], [[2], [0], [-1]],
                 [[1], [-2], [-1]]]
    All_X = [np.array(x) for x in All_X_ori]
    All_trans_fun = ['sgn', 'unipo-sigmoid', 'bipo-sigmoid', 'pie-linear']
    All_learn_rule = ['Hebbian', 'Perception', 'Delta-rule', 'LMS', 'Correlation', 'Winner-take-all', 'Outstar']

    print('样本目录：')
    select_X = select(All_X)

    print('变换函数目录：')
    select_trans_fun = select(All_trans_fun)[0]

    print('学习规则目录：')
    select_learn_rule = select(All_learn_rule)[0]

    return select_X, select_trans_fun, select_learn_rule
    pass


def one_train(x: np.ndarray, w: np.ndarray, lr: float, trans_fun, learn_rule):
    net = np.multiply(w, x)  # 计算网络输入
    out = trans_fun(net)  # 计算网络输出
    dw = learn_rule(x, out, lr)  # 计算权重更新
    w = w - lr * dw  # 更新权重
    return w
    pass


if __name__ == '__main__':
    X, trans_fun, learn_rule = select_data()
    print(X, trans_fun, learn_rule)
