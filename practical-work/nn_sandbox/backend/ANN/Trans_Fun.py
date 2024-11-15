# -*- coding = utf-8 -*-
# @Time : 2024/9/13 下午4:27
# @Author : 李兆堃
# @File : Trans_Fun.py
# @Software : PyCharm
import math


def sgn(net):
    out = 1 if net > 0 else 0 if net == 0 else -1
    return out
    pass


def unipo_sigmoid(net):
    out = 1 / (1 + math.exp(-1 * net))
    return out
    pass


def bipo_sigmoid(net):
    out = (1 - math.exp(-1 * net)) / (1 + math.exp(-1 * net))
    return out
    pass


def pie_linear(net):    # 暂时无题目例子
    pass


def Probabilistic(net):     # 暂时无题目例子
    pass


def None_trans_fun(net):
    return net


if __name__ == '__main__':
    # print(sgn(-3))
    # print(unipo_sigmoid(3.555))
    print(bipo_sigmoid(3.555))
    pass
