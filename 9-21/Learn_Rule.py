# -*- coding = utf-8 -*-
# @Time : 2024/9/14 下午10:36
# @Author : 李兆堃
# @File : Learn_Rule.py
# @Software : PyCharm
import Trans_Fun


def Hebbian(x, out, lr, d, trans_fun):
    return lr * out * x
    pass


def Perception(x, out, lr, d, trans_fun):
    return lr * (d - out) * x
    pass


def Delta_rule(x, out, lr, d, trans_fun):
    f_prime = (1 - out * out) / 2

    return lr * (d - out) * f_prime * x
    pass


def LMS(x, out, lr, d, trans_fun):
    return lr * (d - out) * x
    pass


def Correlation(x, out, lr, d, trans_fun):
    pass


def Winner_take_all(x, out, lr, d, trans_fun):
    pass


def Outstar(x, out, lr):
    pass
