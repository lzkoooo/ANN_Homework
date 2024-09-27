# -*- coding = utf-8 -*-
# @Time : 2024/9/26 上午9:12
# @Author : 李兆堃
# @File : main.py
# @Software : PyCharm
import numpy as np
from Adaline import Adaline


def read_data():
    train_X = []
    train_Y = []
    test_X = []

    with open('train_datasets.csv', 'r', encoding='UTF-8') as f:
        file_data = f.readlines()
        for line in file_data:
            data = [float(i) for i in line.rstrip('\n').split(',')[:-1]]
            train_X.append(np.array(data[:-1]).reshape(-1, 1))
            train_Y.append(float(data[-1]))
    with open('test_datasets.csv', 'r', encoding='UTF-8') as f:
        file_data = f.readlines()
        for line in file_data:
            test_X.append(np.array([float(i) for i in line.rstrip('\n').split(',')]).reshape(-1, 1))

    return train_X, train_Y, test_X


if __name__ == '__main__':
    train_X, train_Y, test_X = read_data()
    W = np.array([[5.0], [1.0]])
    eta = 0.5
    max_epoch = 5000
    print(f' X: {train_X}\n Y: {train_Y}\n W: {W}\n eta: {eta}\n')
    print(f'test_X: {test_X}')

    net = Adaline(train_X, W, train_Y, eta, 'sgn', max_epoch)
    net.fit()
    pred = net.predict(test_X)
    print(pred)
    pass
