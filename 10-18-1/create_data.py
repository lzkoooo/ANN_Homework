# -*- coding = utf-8 -*-
# @Time : 2024/10/14 下午11:31
# @Author : 李兆堃
# @File : create_data.py
# @Software : PyCharm

import numpy as np

# 生成x和y的值在1到5范围的网格，并通过某个函数生成z的值
x = np.linspace(5, 10, 20)
y = np.linspace(5, 10, 20)
x_grid, y_grid = np.meshgrid(x, y)

# 定义一个函数，例如 z = sin(x) + cos(y)
z_grid = np.sin(x_grid) + np.cos(y_grid)

# 将x, y, z数据转换为20条三维数据
data_surface = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

# 提取前20条数据
data_sample = data_surface[np.random.choice(data_surface.shape[0], 20, replace=False)]

# 保存为txt文件
np.savetxt("nn_sandbox/assets/data/sample.txt", data_sample, header="x y z", comments='')

print("数据已保存到 3d_data_sample.txt 文件中")