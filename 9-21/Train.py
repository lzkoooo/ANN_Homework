# -*- coding = utf-8 -*-
# @Time : 2024/9/16 下午8:45
# @Author : 李兆堃
# @File : Train.py
# @Software : PyCharm

import numpy as np
import Learn_Rule
import Trans_Fun


def one_train(x: np.ndarray, d: np.ndarray, W: np.ndarray, lr: float, trans_fun: str, learn_rule: str):
    result = ''
    W_t = W.reshape(1, -1)
    net = np.dot(W_t, x)[0][0]  # 计算网络输入
    # print('网络输入：', net)
    result += '网络输入：' + str(net) + '\n'
    if learn_rule != 'LMS':
        out = getattr(Trans_Fun, trans_fun)(net)  # 计算网络输出
    else:
        out = net
    dw = getattr(Learn_Rule, learn_rule)(x, out, lr, d, trans_fun)  # 计算权重更新
    # print('权重更新：', dw)
    result += '权重更新：' + str(dw) + '\n'
    W += dw  # 更新权重
    # print('更新后的权重：', W_T)
    result += '更新后的权重：' + str(W) + '\n'
    return W, dw, result
    pass


def run_train(X, D, W, lr, trans_fun, learn_rule, max_epoch):
    epoch = 0

    result = ''
    while True:
        dW = None
        for count in range(len(X)):
            # print('\n第{}轮迭代_第{}个样本: '.format(epoch + 1, count + 1))
            result += '\n第{}轮迭代_第{}个样本: \n'.format(epoch + 1, count + 1)
            W, dW, result_2 = one_train(X[count], D[count], W, lr, trans_fun, learn_rule)
            result += result_2
        if np.all(dW == 0):  # 终止条件
            break
        epoch += 1
        if epoch == max_epoch:
            break

    return result
# if __name__ == '__main__':
#     W_T = np.array([[1], [0], [1]]).astype(float)  # 初始权向量
#     lr = 0.25
#     d = [-1, 1]
#
#     X, trans_fun, learn_rule = select_data()
#     print('选择的数据：\n样本：{}\n变换函数：{}\n学习规则：{}'.format(X, trans_fun, learn_rule))
#
#     pass
