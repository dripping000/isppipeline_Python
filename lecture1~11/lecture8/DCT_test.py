import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lena_gray.bmp', 0)

img1 = img.astype('float')

C_temp = np.zeros(img.shape)
dst = np.zeros(img.shape)

m, n = img.shape
N = n
C_temp[0, :] = 1 * np.sqrt(1 / N)

for i in range(1, m):
    for j in range(n):
        C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)
                              ) * np.sqrt(2 / N)

dst = np.dot(C_temp, img1)
dst = np.dot(dst, np.transpose(C_temp))

dst1 = np.log(abs(dst))  # 进行log处理

img_recor = np.dot(np.transpose(C_temp), dst)
img_recor1 = np.dot(img_recor, C_temp)

# 自带方法

img_dct = cv2.dct(img1)  # 进行离散余弦变换

img_dct_log = np.log(abs(img_dct))  # 进行log处理

img_recor2 = cv2.idct(img_dct)  # 进行离散余弦反变换

plt.subplot(231)
plt.imshow(img1, 'gray')
plt.title('original image')
plt.xticks([]), plt.yticks([])

plt.subplot(232)
plt.imshow(dst1)
plt.title('DCT1')
plt.xticks([]), plt.yticks([])

plt.subplot(233)
plt.imshow(img_recor1, 'gray')
plt.title('IDCT1')
plt.xticks([]), plt.yticks([])

plt.subplot(234)
plt.imshow(img, 'gray')
plt.title('original image')

plt.subplot(235)
plt.imshow(img_dct_log)
plt.title('DCT2(cv2_dct)')

plt.subplot(236)
plt.imshow(img_recor2, 'gray')
plt.title('IDCT2(cv2_idct)')

plt.show()
