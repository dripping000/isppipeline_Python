import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage, misc
from scipy import signal
import scipy.ndimage as nd



def laplacian_of_gaussian(image):
    image1=image.copy()
    #高斯滤波
    image2 = ndimage.gaussian_filter(image1, sigma=0.5)
    #拉普拉兹滤波
    result = ndimage.laplace(image2)
    #result = ndimage.gaussian_laplace(image1, sigma=1.2)
    return result


def difference_of_gaussian(image):
    image1 = image.copy()
    #高斯滤波
    image1=ndimage.gaussian_filter(image1, sigma=0.5)
    # image2=image1[0:-1:2,0:-1:2]
    # image2 = ndimage.gaussian_filter(image2, sigma=1.2)
    # size = image1.shape
    # image2 = cv2.resize(image2,size)
    #高斯滤波
    image2 = ndimage.gaussian_filter(image1, sigma=0.5)
    #第一次减第二次
    result=image1-image2
    #result=cv2.subtract(image1,image2)
    return result


if __name__ == '__main__':
    image = plt.imread('lena_gray.bmp', 0)
    image = image.astype(np.int16)

    l = laplacian_of_gaussian(image)
    #负数部分转到的整数
    l = l+np.abs(np.min(l))
    plt.figure()
    plt.imshow(l,'gray',vmax=np.max(l),vmin=np.min(l))
    plt.title('laplacian-of-gaussian')
    plt.axis('off')
    plt.show()

    plt.figure()
    d = difference_of_gaussian(image)
    # 负数部分转到的整数
    d = d + np.abs(np.min(d))
    plt.imshow(d, 'gray',vmax=np.max(d),vmin=np.min(d))
    plt.title('difference-of-gaussian')
    plt.axis('off')
    plt.show()