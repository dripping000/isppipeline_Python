import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''
mono, the image data value is between 0~1
'''
# 黑白全尺寸show
def raw_image_show_fullsize(image, height, width):
    x = width/100
    y = height/100
    plt.figure(num='test', figsize=(x,y))
    plt.imshow(image, cmap='gray', interpolation='bicubic', vmax=1.0,vmin=0.0)
    plt.xticks([]), plt.yticks([])  # 隐藏X轴和Y轴的标记位置和labels
    plt.show()
    print('show')


# 黑白小尺寸show
def raw_image_show_thumbnail(image, height, width):
    x = width / 800
    y = height / 800
    plt.figure(num='test', figsize=(x, y))
    plt.imshow(image, cmap='gray', interpolation='bicubic', vmax=1.0, vmin=0.0)
    plt.xticks([]), plt.yticks([])  # 隐藏X轴和Y轴的标记位置和labels
    plt.show()
    print('show')


def raw_image_show_fakecolor(image, height, width, pattern):
    x = width/100
    y = height/100
    rgb_img = np.zeros(shape=(height, width, 3))
    # GR GB使用的是同一个颜色通道
    R = rgb_img[:,:,0]
    GR = rgb_img[:,:,1]
    GB = rgb_img[:,:,1]
    B = rgb_img[:,:,2]
    if (pattern == "RGGB"):
        R[::2, ::2] = image[::2, ::2]
        GR[::2, 1::2] = image[::2, 1::2]
        GB[1::2, ::2] = image[1::2, ::2]
        B[1::2, 1::2] = image[1::2, 1::2]
    elif (pattern == "GRBG"):
        GR[::2, ::2] = image[::2, ::2]
        R[::2, 1::2] = image[::2, 1::2]
        B[1::2, ::2] = image[1::2, ::2]
        GB[1::2, 1::2]= image[1::2, 1::2]
    elif (pattern == "GBRG"):
        GB[::2, ::2] = image[::2, ::2]
        B[::2, 1::2] = image[::2, 1::2]
        R[1::2, ::2] = image[1::2, ::2]
        GR[1::2, 1::2] = image[1::2, 1::2]
    elif (pattern == "BGGR"):
        B[::2, ::2] = image[::2, ::2]
        GB[::2, 1::2] = image[::2, 1::2]
        GR[1::2, ::2] = image[1::2, ::2]
        R[1::2, 1::2] = image[1::2, 1::2]
    else:
        print("show failed")

    plt.figure(num='test', figsize=(x,y))
    plt.imshow(rgb_img, interpolation='bicubic', vmax=1.0)
    plt.xticks([]), plt.yticks([])  # 隐藏X轴和Y轴的标记位置和labels
    plt.show()
    print('show')


def raw_image_show_3D(image, height, width):
    fig =  plt.figure()
    ax = Axes3D(fig)
    ax = plt.subplot(1, 1, 1, projection='3d')
    X = np.arange(0, width)
    Y = np.arange(0, height)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = image
    # 具体函数方法可用help(function)查看，如：help(ax.plot_surface)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    # ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    plt.show()
    print('show')


def test_case_fullsize():
    b = np.fromfile("RAW_GRBG_plained_4608(9216)x3456_A.raw", dtype ="uint16")
    print("b shape",b.shape)
    print('%#x'%b[0])
    b.shape = [3456, 4608]
    out = b
    out = out/1024.0
    raw_image_show_fullsize(out, 3456, 4608)


def test_case_thumbnail():
    b = np.fromfile("RAW_GRBG_plained_4608(9216)x3456_A.raw", dtype ="uint16")
    print("b shape",b.shape)
    print('%#x'%b[0])
    b.shape = [3456, 4608]
    out = b
    out = out/1024.0
    raw_image_show_thumbnail(out, 3456, 4608)


def test_case_fakecolor():
    b = np.fromfile("RAW_GRBG_plained_4608(9216)x3456_A.raw", dtype ="uint16")
    print("b shape",b.shape)
    print('%#x'%b[0])
    b.shape = [3456, 4608]
    out = b
    out = out/1024.0
    raw_image_show_fakecolor(out,3456, 4608,"GRBG")


def test_case_3D():
    b = np.fromfile("RAW_GRBG_plained_4608(9216)x3456_A.raw", dtype ="uint16")
    print("b shape",b.shape)
    print('%#x'%b[0])
    b.shape = [3456, 4608]
    c = b[0:100,0:120]
    out = c
    out = out/1024.0
    raw_image_show_3D(out, 100, 120)


if __name__ == "__main__":
    print ('This is main of module')

    test_case_fakecolor()
