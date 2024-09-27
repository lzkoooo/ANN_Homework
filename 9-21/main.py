# -*- coding = utf-8 -*-
# @Time : 2024/9/12 下午1:31
# @Author : 李兆堃
# @File : main.py
# @Software : PyCharm

import re
import numpy as np
import Learn_Rule
import Trans_Fun
from Train import run_train


def select(data):
    for count in range(len(data)):
        print('{}、 {}'.format(count + 1, data[count]))
        # print(type(i))
    select_data = [data[int(i) - 1] for i in re.findall(r'\d+', input("请选择第几个:"))]  # i - 1即第一个x下标为0
    # print(select)
    return select_data


def select_data():
    All_X_D = [[[[1], [-2]], 1], [[[0], [1]], 2], [[[2], [3]], -2], [[[1], [1]], -1], [[[2], [1], [-1]], -1], [[[0], [-1], [-1]], 1], [[[2], [0], [-1]], -1],
                 [[[1], [-2], [-1]], 1], [[[1], [-3], [5], [2], [7], [2]], 3]]
    # All_X = [np.array(x) for x in All_X_ori]

    All_trans_fun = ['sgn', 'unipo_sigmoid', 'bipo_sigmoid', 'pie_linear', 'Probabilistic', 'None_trans_fun']
    All_learn_rule = ['Hebbian', 'Perception', 'Delta_rule', 'LMS', 'Correlation', 'Winner_take_all', 'Outstar']

    print('样本目录：')
    select_X = select(All_X_D)
    X = [np.array(x[0]) for x in select_X]
    D = [np.array(x[1]) for x in select_X]
    print('变换函数目录：')
    select_trans_fun = select(All_trans_fun)[0]

    print('学习规则目录：')
    select_learn_rule = select(All_learn_rule)[0]

    return X, D, select_trans_fun, select_learn_rule
    pass


if __name__ == '__main__':
    W = np.array([[float(i)] for i in re.findall(r'-?\d+', input("请输入初始权向量:"))])   # 初始权向量
    lr = float(input("请输入学习率:"))
    max_epoch = 1

    X, D, trans_fun, learn_rule = select_data()
    print('\n\n选择的数据：\n样本：')
    for i in range(len(X)):
        print('x = {}\nd = {}\n'.format(X[i], D[i]))
    print('变换函数：{}\n学习规则：{}\n初始权向量：{}'.format(trans_fun, learn_rule, W))

    run_train(X, D, W, lr, trans_fun, learn_rule, max_epoch)

    pass
