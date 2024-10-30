# -*- coding = utf-8 -*-
# @Time : 2024/10/16 上午10:42
# @Author : 李兆堃
# @File : temp.py
# @Software : PyCharm

# -*- coding: utf-8 -*-
# @Time : 2023/8/24 20:09
# @Author :Muzi
# @File : gradient_1.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt


def func(x, y):
    return x ** 2 + y ** 2


def grad(x, y):
    return 2 * x, 2 * y


# 设置初始点
x_start, y_start = 15, 15
theta = 1e-5  # θ =10**-6 设置阈值
learning_rate = 0.2  # 学习率
# 记录下初始x y值
x_history = [x_start]
y_history = [y_start]
z_history = [func(x_start, y_start)]
# 记录迭代次数
cnt = 0

while True:
    cnt += 1
    # 计算此时梯度
    grad_x, grad_y = grad(x_start, y_start)
    '''计算梯度的范数 即 向量的长度'''
    grad_num = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # 梯度下降 即x-grad*learn_rate
    x_start -= grad_x * learning_rate
    y_start -= grad_y * learning_rate
    # 将梯度下降过程写入列表
    x_history.append(x_start)
    y_history.append(y_start)
    # 判读梯度范数是否小于阈值
    if grad_num < theta:
        print(f"一共迭代了{cnt}次")
        break
    else:
        continue

x = np.linspace(-20, 20, 500)
y = np.linspace(-20, 20, 500)
# 绘制网格线
X, Y = np.meshgrid(x, y)
Z = func(X, Y)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(x_history, y_history, func(np.array(x_history), np.array(y_history)), c='red', marker='o', alpha=1)
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)  # alpha设置透明度
ax.set_title("3D Plot of f(x, y) = x^2/5 + y^2/3")
ax.set_xlabel("x")
ax.set_ylabel("y")
# 记录迭代的痕迹 scatter


cs = fig.add_subplot(122)
cs.contour(X, Y, Z)
cs.set_xlabel('x')
cs.set_ylabel('y')
cs.set_title('contour')

cs.scatter(x_history, y_history, func(np.array(x_history), np.array(y_history)), c='red', marker='.')

plt.show()
