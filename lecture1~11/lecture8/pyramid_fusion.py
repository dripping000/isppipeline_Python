import cv2
import numpy as np
from matplotlib import pyplot as plt
A = plt.imread('apple.jpg')
B = plt.imread('orange.jpg')

# generate Gaussian pyramid for A
G = A.copy()
G = G.astype(np.int16)
gpA = [G]
for i in np.arange(6):     #将苹果进行高斯金字塔处理，总共6级处理
    G = cv2.pyrDown(G)
    gpA.append(G)
# generate Gaussian pyramid for B
G = B.copy()
G = G.astype(np.int16)
gpB = [G]
for i in np.arange(6):  # #将橘子进行高斯金字塔处理，总共6级处理
    G = cv2.pyrDown(G)
    gpB.append(G)
# generate Laplacian Pyramid for A
lpA = [gpA[6]]
for i in np.arange(6,0,-1):    #将苹果进行拉普拉斯金字塔处理，总共6级处理
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)
# generate Laplacian Pyramid for B
lpB = [gpB[6]]
for i in np.arange(6,0,-1):    #将橘子进行拉普拉斯金字塔处理，总共6级处理
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)
# Now add left and right halves of images in each level
#numpy.hstack(tup)
#Take a sequence of arrays and stack them horizontally
#to make a single array.
LS = [] #结果图片的金字塔的存储
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))    #将两个图像的矩阵的左半部分和右半部分拼接到一起
    #ls=la
    LS.append(ls)
# now reconstruct
ls_ = LS[0]   #这里LS[0]为高斯金字塔的最小图片
for i in range(1,7):                        #第一次循环的图像为高斯金字塔的最小图片，依次通过拉普拉斯金字塔恢复到大图像 这里做了6次
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])                #采用金字塔拼接方法的图像
# image with direct connecting each half

real = np.hstack((A[:,:int(cols/2)],B[:,int(cols/2):]))   #直接的左右拼接
plt.figure()
plt.imshow(ls_)
#plt.title(name)
plt.axis('off')
plt.show()
plt.figure()
plt.imshow(real)
#plt.title(name)
plt.axis('off')
plt.show()