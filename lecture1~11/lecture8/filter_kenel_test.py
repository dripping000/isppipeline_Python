import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def kernel_show_3D(image,height,width):
    fig =  plt.figure()
    ax = Axes3D(fig)
    #ax = plt.subplot(1,1,1,projection='3d')
    X = np.arange(0, width)
    Y = np.arange(0, height)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z =image
    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    plt.show()
    print('show')


# 平均滤波器
mean_filter = np.ones((11, 11))

# 高斯滤波器
x = cv2.getGaussianKernel(11,2)
gaussian = x * x.T

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

laplacian = np.array([[0,  1, 0],
                     [1, -4, 1],
                     [0,  1, 0]])

filters = [mean_filter, gaussian, sobel_x, sobel_y, laplacian]
filter_name = ['mean_filter', 'gaussian', 'sobel_x', 'sobel_y', 'laplacian']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [(np.abs(z)) for z in fft_shift]
mag_spectrum[2]= np.log(mag_spectrum[2]+1)
mag_spectrum[3]= np.log(mag_spectrum[3]+1)
mag_spectrum[4]= np.log(mag_spectrum[4]+1)

for i in range(5):
    size=mag_spectrum[i].shape
    kernel_show_3D(mag_spectrum[i], size[0],size[1])
