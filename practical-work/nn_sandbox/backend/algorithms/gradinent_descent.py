
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QLabel, QVBoxLayout, QWidget, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation


class GradientDescentApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('梯度下降方法选择与动态展示')
        layout = QVBoxLayout()

        self.combo = QComboBox(self)
        self.combo.addItems(['BGD', 'SGD', 'MBGD'])

        self.label = QLabel('选择的算法: BGD', self)
        self.button = QPushButton('运行梯度下降', self)
        self.button.clicked.connect(self.run_gradient_descent)

        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')

        layout.addWidget(self.combo)
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def run_gradient_descent(self):
        algorithm = self.combo.currentText()
        self.label.setText(f'选择的算法: {algorithm}')
        self.animate(algorithm)

    def animate(self, algorithm):
        # 清空之前的数据
        self.ax.clear()

        # 定义梯度下降参数
        learning_rate = 0.1
        epoch = 50
        num_iter = 4
        start_point = np.array([4, 4]).astype(np.float64)  # 起始点


        self.x_data = []
        self.y_data = []
        self.z_data = []

        def gradient(point):
            return np.array([2 * point[0], 2 * point[1]])  # 求梯度

        for _ in range(epoch):
            sgd_list = []
            for _ in range(num_iter):
                grad = gradient(start_point)
                if algorithm == 'BGD':
                    grad += grad
                elif algorithm == 'SGD':
                    sgd_list.append(grad)
                else:
                    adj = learning_rate * grad
                    start_point -= adj
                    self.x_data.append(start_point[0])
                    self.y_data.append(start_point[1])
                    self.z_data.append(start_point[0] ** 2 + start_point[1] ** 2)  # 对应的Z值

            if algorithm == 'BGD':
                adj = learning_rate * grad
            elif algorithm == 'SGD':
                adj = learning_rate * sgd_list[np.random.randint(0, len(sgd_list))]
            if algorithm != 'MBGD':
                start_point -= adj
                self.x_data.append(start_point[0])
                self.y_data.append(start_point[1])
                self.z_data.append(start_point[0] ** 2 + start_point[1] ** 2)  # 对应的Z值
        # 生成3D图像
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = X ** 2 + Y ** 2  # 假设的二次函数

        self.ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

        self.ani = FuncAnimation(self.fig, self.update_plot, frames=np.arange(0, epoch), interval=500,
                                 blit=False)
        self.canvas.draw()

    def update_plot(self, frame):
        self.ax.scatter(self.x_data[frame], self.y_data[frame], self.z_data[frame], color='r', s=100)
        return self.ax,