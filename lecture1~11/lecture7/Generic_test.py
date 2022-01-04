from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage


def test_generic_filter():
    def filter2d(footprint_elements, weights):
        return (weights * footprint_elements).sum()

    im = plt.imread("eight.tif")  # 在这里读取图片
    footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    footprint_size = np.count_nonzero(footprint) #genric_filter 元素只有5个,所以这里只有非0的元素
    weights = np.ones(footprint_size) / footprint_size #归一化参数

    std = ndimage.generic_filter(im, filter2d, footprint=footprint,extra_arguments=(weights,))
    plt.imshow(im,cmap = 'gray')
    plt.show()
    plt.imshow(std,cmap = 'gray')
    plt.show()

def test_generic_filter2():
    def mid_pass(P):     #函数中可以做很多卷积做不了的操作
        return np.median(P)

    im = plt.imread("eight.tif")  # 在这里读取图片
    footprint = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) #3X3区域做滤波
    std = ndimage.generic_filter(im, mid_pass, footprint=footprint)
    plt.imshow(im,cmap = 'gray')
    plt.show()
    plt.imshow(std,cmap = 'gray')
    plt.show()






if __name__ == "__main__":
    print ('This is main of module')
    test_generic_filter()
