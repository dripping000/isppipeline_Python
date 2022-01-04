import numpy as np
import matplotlib.pyplot as plt

#均值滤波函数
def mean_filter(im,x, y, step):
    sum_s = 0
    for k in range(-int(step/2),int(step/2)+1):
        for m in range(-int(step/2),int(step/2)+1):
            sum_s += im[x+k][y+m] / (step*step)
    return sum_s

medStep=3
meaStep=3
im = plt.imread("eight.tif")  # 在这里读取图片
plt.imshow(im,cmap = 'gray')
plt.show()
image_size=np.shape(im)
im_copy_mea=np.zeros(image_size)
im_copy_iir=im.copy()

#FIR
for i in range(int(meaStep/2),im.shape[0]-int(meaStep/2)):
    for j in range(int(meaStep/2),im.shape[1]-int(meaStep/2)):
        im_copy_mea[i][j] = mean_filter(im,i, j, meaStep)           #使用源图片作为每次的输入
plt.imshow(im_copy_mea,cmap = 'gray')                               #显示卷积后的图片
plt.show()

#IIR
for i in range(int(meaStep/2),im.shape[0]-int(meaStep/2)):
    for j in range(int(meaStep/2),im.shape[1]-int(meaStep/2)):
        im_copy_iir[i][j] = mean_filter(im_copy_iir,i, j, meaStep)  #滤波的结果图片作为每次的输入

plt.imshow(im_copy_iir,cmap = 'gray')                               #显示卷积后的图片
plt.show()