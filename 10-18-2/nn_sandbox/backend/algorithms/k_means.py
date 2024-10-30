import numpy as np

from ..utils import dist


class KMeans:
    def __init__(self, num):
        self.clusters = [Cluster() for _ in range(num)]

    def fit(self, dataset: np.ndarray, search_times=100, tolerance=0.005):
        # initial_centers = dataset[np.random.choice(
        #     dataset.shape[0], len(self.clusters), replace=False
        # ), :-1]
        initial_centers = np.random.uniform(-4.0, 4.0, (len(self.clusters), 1))     # 初始数据中心

        for cluster, init_center in zip(self.clusters, initial_centers):
            cluster.center = init_center

        for _ in range(search_times):
            self.groupify(dataset)
            for cluster in self.clusters:
                cluster.update_center()
            if max(cluster.diff for cluster in self.clusters) < tolerance:
                break

        return self.clusters

    def groupify(self, dataset):
        for data in dataset:
            distances = {dist(cluster.center, data[:-1]): cluster
                         for cluster in self.clusters}
            distances[min(distances)].member.append(data[:-1])


class Cluster:
    def __init__(self):
        self.center: np.ndarray = None
        self.member = []
        self.diff = 0

    @property
    def avg_distance(self):
        if not self.member:
            # return 0
            return np.random.uniform(0.1, 0.3)  # 初始扩展常数
        return sum(dist(self.center, data) for data in self.member) / len(self.member)

    def update_center(self):
        if not self.member:
            return
        new_center = sum(self.member) / len(self.member)
        self.diff = max(abs(new_center - self.center))
        self.center = new_center
