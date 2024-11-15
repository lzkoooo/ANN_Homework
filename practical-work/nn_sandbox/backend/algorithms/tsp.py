# -*- coding = utf-8 -*-
# @Time : 2024/10/29 13:12
# @Author : 李兆堃
# @File : chnn.py
# @Software : PyCharm
import sys
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QColor, QPainter
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QGraphicsScene, QGraphicsView, \
    QDialog


class HopfieldTSP:
    def __init__(self, cities, alpha=200, beta=100):
        self.cities = np.array(cities)
        self.n_cities = len(cities)
        self.alpha = alpha
        self.beta = beta
        self.u = 0.1 * np.log(self.n_cities - 1) + 2 * np.random.rand(self.n_cities, self.n_cities) - 1
        self.v = np.zeros_like(self.u)
        self.update_v()
        self.route = None
        self.energy = None

    def distance_matrix(self):
        dist = np.linalg.norm(self.cities[:, np.newaxis] - self.cities, axis=2)
        return dist

    @property
    def total_distance(self):
        dist = 0
        for i in range(len(self.route) - 1):
            dist += np.linalg.norm(self.cities[self.route[i] - 1] - self.cities[self.route[i + 1] - 1])
        return dist

    def update_v(self):
        self.v = 0.5 * (1 + np.tanh(self.u / 0.1))

    def update(self):
        dist = self.distance_matrix()
        a = np.sum(self.v, axis=1, keepdims=True) - 1
        b = np.sum(self.v, axis=0, keepdims=True) - 1
        a = np.repeat(a, self.v.shape[1], axis=1)
        b = np.repeat(b, self.v.shape[0], axis=0)

        c1 = self.v[:, 1:self.n_cities]
        c0 = np.zeros((self.n_cities, 1))
        c0[:, 0] = self.v[:, 0]
        c = np.concatenate((c1, c0), axis=1)
        c = np.dot(dist, c)
        delta_u = -self.alpha * (a + b) - self.beta * c
        self.u += delta_u * 0.0001
        self.update_v()
        self.energy = self.energy_function(dist)
        self.route = self.get_route()
        print(f"current energy:{self.energy}")

    def get_route(self):
        route = np.argmax(self.v, axis=1) + 1
        route = np.append(route, route[0])
        return route

    def energy_function(self, dist):
        t1 = np.sum(np.power(np.sum(self.v, axis=0) - 1, 2))
        t2 = np.sum(np.power(np.sum(self.v, axis=1) - 1, 2))
        idx = [i for i in range(1, self.n_cities)] + [0]
        vt = self.v[:, idx]
        t3 = dist * vt
        t3 = np.sum(np.sum(np.multiply(self.v, t3)))
        e = 0.5 * (self.alpha * (t1 + t2) + self.beta * t3)
        return e


class TSPWindow(QDialog):
    def __init__(self, cities):
        super().__init__()
        self.setWindowTitle("TSP旅行商问题")
        self.setGeometry(1100, 600, 800, 800)
        self.cities = cities
        self.hopfield_net = HopfieldTSP(cities)
        layout = QVBoxLayout()

        self.start_button = QPushButton("计算最短路径", self)
        self.start_button.clicked.connect(self.calculate_path)

        self.result_label = QLabel("", self)

        self.graphics_view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)

        layout.addWidget(self.start_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.graphics_view)
        self.setLayout(layout)

    def calculate_path(self):
        best_route = []
        min_dist = None
        for _ in range(40000):
            self.hopfield_net.update()
            route = self.hopfield_net.route
            if len(np.unique(route)) == len(route[:-1]):
                dist = self.hopfield_net.total_distance
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    best_route = route
        self.paint_route(best_route - 1)
        self.result_label.setText(f"最短路径: {best_route}, 最短距离：{min_dist:.2f}")

    def paint_route(self, route):
        self.scene.clear()
        pen = QPen(QColor(0, 0, 255), 2)
        for i in range(len(route) - 1):
            self.scene.addLine(self.cities[route[i]][0] * 600, (1 - self.cities[route[i]][1]) * 400,
                self.cities[route[i + 1]][0] * 600, (1 - self.cities[route[i + 1]][1]) * 400,
                pen
            )
        for city in self.cities:
            self.scene.addEllipse(city[0] * 600 - 5, (1 - city[1]) * 400 - 5, 10, 10, QPen(Qt.black), QColor(255, 0, 0)
            )


if __name__ == '__main__':
    cities = [
        [0.1, 0.6], [0.8, 0.3], [0.5, 0.6], [0.4, 0.9],
        [0.7, 0.1], [0.5, 0.5], [0.7, 0.3], [0.8, 0.8],
        [0.2, 0.9], [0.4, 0.1]
    ]
    app = QApplication(sys.argv)
    window = TSPWindow(cities)
    window.show()
    sys.exit(app.exec_())
