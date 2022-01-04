import cv2
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

#原始曲面曲面
def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
#100X100的矩阵
grid_x, grid_y = np.mgrid[0:99:100j, 0:99:100j] #实数的时候指间隔,虚数指取多少个点
#
point_x, point_y = np.mgrid[0:100:3, 0:100:3] #34个点
#拉成一维数组
point_x=point_x.flatten()
point_y=point_y.flatten()
#有多少个元素
len_x=point_x.shape
points = np.zeros(shape=(len_x[0], 2))
points[:,0]=point_x
points[:,1]=point_y
#34X34矩阵
values = func(points[:,0]/100, points[:,1]/100)
#grid实现插值
grid_z0 = interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = interpolate.griddata(points, values, (grid_x, grid_y), method='cubic')


plt.subplot(221)
plt.imshow(func(grid_x/100, grid_y/100), extent=(0,1,0,1), origin='lower')
plt.plot(points[:,0]/100, points[:,1]/100, 'k.', ms=1) #黑点标出原来的点
plt.title('Original')
plt.subplot(222)
plt.imshow(grid_z0, extent=(0,1,0,1), origin='lower')
plt.title('Nearest')
plt.subplot(223)
plt.imshow(grid_z1, extent=(0,1,0,1), origin='lower')
plt.title('Linear')
plt.subplot(224)
plt.imshow(grid_z2, extent=(0,1,0,1), origin='lower')
plt.title('Cubic')
plt.gcf().set_size_inches(6, 6)
plt.show()
values.shape=(34,34)
#opencv的实现
newimg1 = cv2.resize(values, (100, 100), interpolation=cv2.INTER_NEAREST)# 放大,象素关系重采样
newimg2 = cv2.resize(values, (100, 100), interpolation=cv2.INTER_LINEAR)
newimg3 = cv2.resize(values, (100, 100), interpolation=cv2.INTER_CUBIC)
newimg4 = cv2.resize(values, (100, 100), interpolation=cv2.INTER_AREA)# 和最邻近并不同
newimg5 = cv2.resize(values, (100, 100), interpolation=cv2.INTER_LANCZOS4)# 和最邻近并不同

plt.subplot(231)
plt.imshow(func(grid_x/100, grid_y/100), extent=(0,1,0,1), origin='lower')
plt.plot(points[:,0]/100, points[:,1]/100, 'k.', ms=1)
plt.title('Original')
plt.subplot(232)
plt.imshow(newimg1, extent=(0,1,0,1), origin='lower')
plt.title('Nearest')
plt.subplot(233)
plt.imshow(newimg2, extent=(0,1,0,1), origin='lower')
plt.title('Linear')
plt.subplot(234)
plt.imshow(newimg3, extent=(0,1,0,1), origin='lower')
plt.title('Cubic')
plt.subplot(235)
plt.imshow(newimg4, extent=(0,1,0,1), origin='lower')
plt.title('Area')
plt.subplot(236)
plt.imshow(newimg5, extent=(0,1,0,1), origin='lower')
plt.title('LANCZOS4')
plt.gcf().set_size_inches(12, 8)
plt.show()



x=np.arange(0,100,3)
y=np.arange(0,100,3)
x_new=np.arange(0,100,1)
y_new=np.arange(0,100,1)
newfunc = interpolate.interp2d(x,y, values, kind='linear')
interp2d_1=newfunc(x_new,y_new)
newfunc = interpolate.interp2d(x,y, values, kind='cubic')
interp2d_2=newfunc(x_new,y_new)
newfunc = interpolate.interp2d(x,y, values, kind='quintic')
interp2d_3=newfunc(x_new,y_new)

plt.subplot(221)
plt.imshow(func(grid_x/100, grid_y/100), extent=(0,1,0,1), origin='lower')
plt.plot(points[:,0]/100, points[:,1]/100, 'k.', ms=1)
plt.title('Original')
plt.subplot(222)
plt.imshow(interp2d_1, extent=(0,1,0,1), origin='lower')
plt.title('Linear')
plt.subplot(223)
plt.imshow(interp2d_2, extent=(0,1,0,1), origin='lower')
plt.title('Cubic')
plt.subplot(224)
plt.imshow(interp2d_3, extent=(0,1,0,1), origin='lower')
plt.title('Quintic')
plt.gcf().set_size_inches(6, 6)
plt.show()


