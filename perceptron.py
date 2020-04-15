import numpy as np
from matplotlib import pyplot as plt


class Perceptron:

    def __init__(self, train_x, train_y, eta=1):
        # 超参数：步长
        self.eta = eta
        # 初始化参数w, b
        self.w = np.zeros(train_x.shape[1])
        self.b = 0
        # 保存训练集
        self.train_x = train_x
        self.train_y = train_y
        # 迭代次数，用来评估模型训练速度
        self.iteration_num = 0

    def train(self):
        # 取出所有误分类点
        wrong_points = np.where(self.train_y * (np.dot(self.train_x, self.w.T) + self.b) <= 0)
        self.iteration_num = 0
        # 梯度下降直至没有误分类点
        while wrong_points[0].any():
            self.iteration_num += 1
            # 打乱顺序，随机取出一个误分类点
            np.random.shuffle(wrong_points[0])
            x = self.train_x[wrong_points[0][0]]
            y = self.train_y[wrong_points[0][0]]
            self.w += self.eta * x * y
            self.b += self.eta * y
            wrong_points = np.where(self.train_y * (np.dot(self.train_x, self.w.T) + self.b) <= 0)

    # 预测函数
    def predict(self, x):
        if np.dot(x, self.w.T) + self.b > 0:
            return 1
        else:
            return -1

    # 绘制分类结果
    def draw(self):
        # 训练集点，正例蓝色，负例红色
        x1 = self.train_x[np.where(self.train_y > 0)][:, 0]
        y1 = self.train_x[np.where(self.train_y > 0)][:, 1]
        x2 = self.train_x[np.where(self.train_y < 0)][:, 0]
        y2 = self.train_x[np.where(self.train_y < 0)][:, 1]
        # 超平面S
        x3 = np.arange(0, 6, 0.1)
        y3 = (self.w[0] * x3 + self.b) / (-self.w[1])
        plt.plot(x1, y1, 'bo', x2, y2, 'ro', x3, y3)
        plt.show()


def main():
    # 数据集
    train_x = np.array([[1, 0.2], [2, 1.1], [3, 2.3], [4, 0.25], [0.6, 0.1], [3.3, 3.1], [1.5, 1.1], [3.2, 2.9], [2.5, 1.2], [1.8, 1.4], [2.5, 4], [2.7, 3], [0.3, 2.2], [1.3, 2.1], [1.7, 2.0], [2.6, 2.9], [4.2, 5.2], [1.6, 3.1], [0.5, 1.15], [2.2, 4]])
    train_y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    perceptron = Perceptron(train_x, train_y)
    perceptron.train()
    perceptron.draw()
    print("迭代次数：", perceptron.iteration_num)


if __name__ == "__main__":
    main()
