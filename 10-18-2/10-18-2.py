import numpy as np


def create_datasets():
     p = np.arange(-1, 1.1, 0.1)
     y = [-0.9602, -0.5770, -0.0729, 0.3771, 0.6405, 0.6600, 0.4609, 0.1336, -0.2013, -0.4344, -0.5000, -0.3939, -0.1647, 0.0988, 0.3072, 0.3960, 0.3449, 0.1816, 0.0312, -0.2189, -0.3201]

     data = np.column_stack((p, y, np.ones(len(p))))
     np.savetxt('nn_sandbox/assets/data/18-2.txt', data, fmt='%.4f', delimiter=' ')


if __name__ == '__main__':
    create_datasets()