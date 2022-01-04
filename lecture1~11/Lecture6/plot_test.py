import numpy as np
import matplotlib.pyplot as plt
def Fun(x,a1,a2,a3):                   # 定义拟合函数形式
    return a1*x**2+a2*x+a3
def main():
    x = np.linspace(-10, 10, 100)  # 创建时间序列
    a1, a2, a3 = [-2, 5, 10]  # 原始数据的参数
    noise = np.random.randn(len(x))  # 创建随机噪声
    y = Fun(x, a1, a2, a3) + noise * 2  # 加上噪声的序列
    plt.plot(x, y)
    para = np.polyfit(x, y, deg=2)

    y_fitted = Fun(x, para[0], para[1], para[2])
    plt.figure
    plt.plot(x, y, 'ro', label='Original curve')
    plt.plot(x, y_fitted, '-b', label='Fitted curve')
    plt.legend()
    plt.show()
    print(para)


if __name__ == '__main__':
    main()