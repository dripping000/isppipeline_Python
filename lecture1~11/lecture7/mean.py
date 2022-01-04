import numpy as np
import matplotlib.pyplot as plt


#中值滤波
def m_filter(im,x, y, step):
    sum_s=[]
    for k in range(-int(step/2),int(step/2)+1):
        for m in range(-int(step/2),int(step/2)+1):
            sum_s.append(im[x+k][y+m]) #所有的元素放到list里
    sum_s.sort()#列表排序
    return sum_s[(int(step*step/2)+1)]  #返回list中中间位置的一个

#均值滤波
def mean_filter(im,x, y, step):
    sum_s = 0
    for k in range(-int(step/2),int(step/2)+1):
        for m in range(-int(step/2),int(step/2)+1):
            sum_s += im[x+k][y+m] / (step*step)
    return sum_s


#滤波窗口大小
medStep=3
meaStep=3
im = plt.imread("eight.tif")  # 在这里读取图片
plt.imshow(im,cmap = 'gray')
image_size=np.shape(im)
im_copy_med=np.zeros(image_size)
im_copy_mea=np.zeros(image_size)
#不推荐使用For For形式处理
for i in range(int(medStep/2),im.shape[0]-int(medStep/2)): #
    for j in range(int(medStep/2),im.shape[1]-int(medStep/2)):
        im_copy_med[i][j] = m_filter(im,i, j, medStep)
plt.imshow(im_copy_med,cmap = 'gray')                                 #显示图片
plt.show()

#不推荐使用For For形式处理
for i in range(int(meaStep/2),im.shape[0]-int(meaStep/2)):
    for j in range(int(meaStep/2),im.shape[1]-int(meaStep/2)):
        im_copy_mea[i][j] = mean_filter(im,i, j, meaStep)
plt.imshow(im_copy_mea,cmap = 'gray')
plt.show()