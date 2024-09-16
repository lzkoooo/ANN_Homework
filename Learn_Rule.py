# -*- coding = utf-8 -*-
# @Time : 2024/9/16 下午10:36
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
    f = getattr(Trans_Fun, trans_fun)
    h = 1e-5
    f_prime = (f(out + h) - f(out - h)) / (2 * h)

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
