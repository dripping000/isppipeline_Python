import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np

img = plt.imread("kodim19.png")

# 中值滤波
img_medianBlur = cv2.medianBlur(img, 5)
plt.imshow(img_medianBlur)                                     #显示卷积后的图片
plt.show()
# 均值滤波
img_Blur = cv2.blur(img, (5, 5))
plt.imshow(img_Blur)                                     #显示卷积后的图片
plt.show()
# 高斯滤波
img_GaussianBlur = cv2.GaussianBlur(img, (7, 7), 0)
plt.imshow(img_GaussianBlur)                                     #显示卷积后的图片
plt.show()
# 高斯双边滤波
img_bilateralFilter = cv2.bilateralFilter(img, 40, 75, 75)
plt.imshow(img_bilateralFilter)                                     #显示卷积后的图片
plt.show()