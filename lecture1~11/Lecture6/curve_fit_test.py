import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def Fun(x, a1, a2, a3):  # 定义拟合函数形式
    return a1 * x ** 2 + a2 * x + a3


def error(p, x, y):  # 拟合残差
    return Fun(p, x) - y


def main():
    x = np.linspace(-10, 10, 100)  # 创建时间序列
    a1, a2, a3 = [-2, 5, 10]  # 原始数据的参数
    noise = np.random.randn(len(x))  # 创建随机噪声
    y = Fun(x, a1, a2, a3) + noise * 2  # 加上噪声的序列
    para, pcov = curve_fit(Fun, x, y)
    y_fitted = Fun(x, para[0], para[1], para[2])  # 画出拟合后的曲线

    plt.figure
    plt.plot(x, y, 'r', label='Original curve')
    plt.plot(x, y_fitted, '-b', label='Fitted curve')
    plt.legend()
    plt.show()
    print(para)


if __name__ == '__main__':
    main()