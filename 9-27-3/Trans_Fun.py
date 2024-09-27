# -*- coding = utf-8 -*-
# @Time : 2024/9/26 上午8:27
# @Author : 李兆堃
# @File : Trans_Fun.py
# @Software : PyCharm
import math


class Trans_Fun:
    def __init__(self, trans_fun):
        self.tf_name = trans_fun

    def get_out(self, net):
        out = net
        if self.tf_name == 'sgn':
            out = 1 if net > 0 else -1
        return out
        pass


if __name__ == '__main__':
    # print(sgn(-3))
    # print(unipo_sigmoid(3.555))
    print(bipo_sigmoid(3.555))
    pass
