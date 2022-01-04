import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


def Fun(p, x):  # 定义拟合函数形式
    a1, a2, a3 = p
    return a1 * x ** 2 + a2 * x + a3


def error(p, x, y):  # 拟合残差
    return Fun(p, x) - y


def main():
    x = np.linspace(-10, 10, 100)  # 创建时间序列
    p_value = [-2, 5, 10]  # 原始数据的参数
    noise = np.random.randn(len(x))  # 创建随机噪声
    y = Fun(p_value, x) + noise * 2  # 加上噪声的序列
    p0 = [0.1, -0.01, 100]  # 拟合的初始参数设置
    para = leastsq(error, p0, args=(x, y))  # 进行拟合
    y_fitted = Fun(para[0], x)  # 画出拟合后的曲线

    plt.figure
    plt.plot(x, y, 'r', label='Original curve')
    plt.plot(x, y_fitted, '-b', label='Fitted curve')
    plt.legend()
    plt.show()
    print(para[0])


if __name__ == '__main__':
    main()