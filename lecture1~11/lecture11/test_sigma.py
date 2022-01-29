import numpy as np
import matplotlib.pyplot as plt


def demo1():
    mu, sigma = 0, 1
    sampleNo = 100000

    # mu是中值,sigma是正态分布的系数,smapleNo是样本数量
    np.random.seed(0)
    s = np.random.normal(mu, sigma, sampleNo)

    plt.hist(s, bins=100, density=True)
    plt.show()


# 正态分布的概率密度函数。可以理解成 x 是 mu（均值）和 sigma（标准差）的函数
def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (np.sqrt(2*np.pi) * sigma)
    return pdf


def demo2():
    mu = 32.86
    sigma = 1.93
    # Python实现正态分布
    # 绘制正态分布概率密度函数
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 50)
    y_sig = np.exp(-(x - mu) ** 2 /(2* sigma **2))/(np.sqrt(2*np.pi)*sigma)
    plt.plot(x, y_sig, "r-", linewidth=2)
    plt.xlabel('Frequecy')
    plt.ylabel('Latent Trait')
    plt.title('Normal Distribution: $\mu = %.2f, $sigma=%.2f'%(mu,sigma))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    demo1()
    demo2()
