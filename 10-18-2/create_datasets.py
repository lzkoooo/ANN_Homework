import numpy as np


def function(x):
    return 1.1 * (1 - x + 2 * x ** 2) * np.exp(-x ** 2 / 2)


def create_datasets():
    x = np.random.uniform(-4, 4, 100)
    noise = np.random.normal(0, 0.1, 100)
    y_samples = function(x) + noise
    datasets = np.column_stack((x, y_samples, np.ones(100)))
    return datasets
    pass


def save_datasets(datasets):
    np.savetxt('nn_sandbox/assets/data/5_3.txt', datasets, fmt='%.3f', delimiter=' ')
    pass


if __name__ == '__main__':
    datasets = create_datasets()
    save_datasets(datasets)
