import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lena_gray.bmp',0) #直接读为灰度图像
# Numpy FFT2 是一个二维的复数数组
f = np.fft.fft2(img)
#把低频移到了中心
fshift = np.fft.fftshift(f)
s1 = np.log(np.abs(f))#取对数的目的为了将数据变化到较小的范围（比如0-255）
s2 = np.log(np.abs(fshift))
ph_f = np.angle(f)
ph_fshift = np.angle(fshift)
plt.subplot(221),plt.imshow(s1,'gray'),plt.title('original FFT')
plt.subplot(222),plt.imshow(s2,'gray'),plt.title('center')
plt.subplot(223),plt.imshow(ph_f,'gray'),plt.title('original')
plt.subplot(224),plt.imshow(ph_fshift,'gray'),plt.title('center')
plt.show()