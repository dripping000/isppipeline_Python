import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import color_utils as color
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
def plot_lut(lut,LUT_SIZE):
    fig =  plt.figure()
    ax = Axes3D(fig)
    ax = plt.subplot(1,1,1,projection='3d')
    X = np.arange(0, LUT_SIZE)
    Y = np.arange(0, LUT_SIZE)
    X, Y = np.meshgrid(X, Y)
    lut=lut.reshape(LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)
    Z =image
    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.plot_wireframe(X, Y,lut[:,:,0,1], rstride=10, cstride=10)
    plt.show()
    print('show')


def load_lut(path,LUT_SIZE):
    lut = np.zeros((LUT_SIZE**3, 3))
    with open(path, 'r') as f:
        for num, l in enumerate(f.readlines()[-LUT_SIZE**3:]):
            l = np.array(l.strip().split(' ')).astype(np.float32)
            lut[num] = l
    return lut

def apply_3DLUT(img,lut,LUT_SIZE,maxvalue):
    x = np.arange(0, LUT_SIZE)
    interpolation_func = RegularGridInterpolator(
        (x, x, x), lut.reshape(LUT_SIZE, LUT_SIZE, LUT_SIZE, 3), method="linear")

    new_image = interpolation_func((img/maxvalue)*(LUT_SIZE-1)) #归一插值
    new_image *= maxvalue #反归一
    new_image=np.clip(new_image,0,maxvalue)
    return new_image


if __name__ == "__main__":
   #read image
   # read_image
    maxvalue = 255
    LUT_SIZE=17
    pattern = "GRBG"
    image = plt.imread('kodim19.png')
    if (np.max(image) <= 1):
       image = image * maxvalue
    lut = load_lut("Levine.cube", LUT_SIZE)
    plot_lut(lut,LUT_SIZE)
    new_image=apply_3DLUT(image,lut,LUT_SIZE,maxvalue)
    color.rgb_show(image/255)
    color.rgb_show(new_image/255)

