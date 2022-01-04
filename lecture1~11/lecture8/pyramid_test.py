import numpy as np
import cv2
from matplotlib import pyplot as plt
def gaussian(ori_image, down_times=3):
    # 1：添加第一个图像为原始图像
    temp_gau = ori_image.copy()
    gaussian_pyramid = [temp_gau]
    for i in range(down_times):
        # 2：下采样次数的金字塔
        temp_gau = cv2.pyrDown(temp_gau)
        gaussian_pyramid.append(temp_gau)
    return gaussian_pyramid

def laplacian(gaussian_pyramid, up_times=3):
    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(up_times, 0, -1):
        # 上采样的次数
        temp_pyrUp = cv2.pyrUp(gaussian_pyramid[i]) #上采样
        temp_lap = cv2.subtract(gaussian_pyramid[i-1], temp_pyrUp)#和高斯金字塔对应层减值就得到了对应的拉普拉兹金字塔
        laplacian_pyramid.append(temp_lap)
    return laplacian_pyramid

if __name__ == '__main__':
    image = plt.imread('lena_gray.bmp', 0)
    image = image.astype(np.int16)
    plt.figure()
    plt.imshow(image, 'gray')
    plt.title('Original')
    plt.show()
    times=3 #下采样次数为3
    gaussian_pyramid=gaussian(image,times)
    for i in range(times+1):
        plt.figure()
        plt.imshow(gaussian_pyramid[i], 'gray')
        name="gaussian_pyraid" + str(i + 1)+".bmp"  #拼接文件名
        plt.imsave(name,gaussian_pyramid[i],cmap='gray' ) #存储
        plt.title('name')
        plt.axis('off')
        plt.show()

    laplacian_pyramid=laplacian(gaussian_pyramid,times) #用高斯金字塔生成拉普拉兹金字塔
    for i in range(times+1):
        plt.figure()
        temp_image=laplacian_pyramid[i].copy()
        if(np.min(laplacian_pyramid[i])<0):
             temp_image=laplacian_pyramid[i]+np.abs(np.min(laplacian_pyramid[i])) #调整负数到整数为了显示
        plt.imshow(temp_image, 'gray')
        name="laplacian_pyramid" + str(i + 1)+".bmp"
        plt.imsave(name,laplacian_pyramid[i],cmap='gray' )
        plt.title(name)
        plt.axis('off')
        plt.show()