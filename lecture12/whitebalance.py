
import raw_image as raw
import raw_image_show
import read_plained_raw as plained_raw
import numpy as np
import matplotlib.pyplot as plt
import color_utils as color
from skimage import filters
from scipy import signal

import cv2


# RGB转ycbcr
def rgb2ycbcr(R, G, B):
    size = R.shape
    im = np.empty((size[0], size[1], 3), dtype=np.float32)
    im[:,:,0] = R
    im[:,:,1] = G
    im[:,:,2] = B

    xform = np.array([[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return ycbcr  # np.uint8(ycbcr)


def grey_world(R,G,B):
    mu_r = np.average(R)
    mu_g = np.average(G)
    mu_b = np.average(B)

    illum_max = np.maximum(mu_r, mu_g)
    illum_max = np.maximum(illum_max, mu_b)

    R_gain = illum_max / mu_r
    G_gain = illum_max / mu_g
    B_gain = illum_max / mu_b

    return R_gain, G_gain, B_gain


def auto_threshold(R, G, B):
    # 转换到ycbcr
    x, y = R.shape
    ycc = rgb2ycbcr(R, G, B)
    Lu = ycc[:, :, 0]
    Cb = ycc[:, :, 1]
    Cr = ycc[:, :, 2]

    # 计算cbcr的均值 Mb Mr
    Mb = np.mean(Cb)
    Mr = np.mean(Cr)
    # 计算Cb Cr的方差
    Db = np.sum(np.abs(Cb - Mb)) / (x * y)
    Dr = np.sum(np.abs(Cr - Mr)) / (x * y)

    # 根据阈值的要求提取出near- white区域的像素点
    b1 = Cb - (Mb + Db * np.sign(Mb))
    b2 = Cr - (Mr + Dr * np.sign(Mr))
    itemindex = np.where((np.abs(b1) < (1.5 * Db)) & (np.abs(b2) < (1.5 * Dr)))
    L_NW = Lu[itemindex]

    # 像素点排序
    L_NW_sorted = np.sort(L_NW)
    # 提取前10%的点作为参考白点
    count = L_NW.shape
    nn = round(count[0]*9/10)
    threshold = L_NW_sorted[nn-1]
    itemindex2 = np.where(L_NW>=threshold)

    # 提取参考白点的RGB三信道的值
    R_NW = R[itemindex]
    G_NW = G[itemindex]
    B_NW = B[itemindex]
    R_selected = R_NW[itemindex2]
    G_selected = G_NW[itemindex2]
    B_selected = B_NW[itemindex2]

    # 计算对应点的RGB均值
    mu_r = np.mean(R_selected)
    mu_g = np.mean(G_selected)
    mu_b = np.mean(B_selected)
    # 计算增益
    illum_max = np.maximum(mu_r, mu_g)
    illum_max = np.maximum(illum_max, mu_b)

    R_gain = illum_max / mu_r
    G_gain = illum_max / mu_g
    B_gain = illum_max / mu_b

    return R_gain, G_gain, B_gain


def set_border(im, width, method):
    # sets border to either zero method=0, or method=1 to average
    hh, ll = im.shape
    im[0:width, :] = method
    im[hh-width:hh, :] = method
    im[:, 0:width] = method
    im[:, ll-width:ll] = method
    return im


def dilation33(im):
    hh, ll = im.shape
    out1 = np.zeros((hh, ll))
    out2 = np.zeros((hh, ll))
    out3 = np.zeros((hh, ll))

    # H方向扩展上下像素
    out1[0:hh-1, :] = im[1:hh, :]
    out1[hh-1, :] = im[hh-1, :]
    out2 = im
    out3[0, :] = im[0, :]
    out3[1:hh, :] = im[0:hh-1, :]
    out_max = np.maximum(out1, out2)
    out_max = np.maximum(out_max, out3)

    # W方向扩展上下像素
    out1[:, 0:ll-1] = out_max[:, 1:ll]
    out1[:, ll-1] = out_max[:, ll-1]
    out2 = out_max
    out3[:, 0] = out_max[:, 0]
    out3[:, 1:ll] = out_max[:, 0:ll-1]
    out_max=np.maximum(out1, out2)
    out_max=np.maximum(out_max, out3)
    return out_max


def gDer(f, sigma, iorder, jorder):
    break_off_sigma = 3.
    H, W = f.shape
    filtersize = np.floor(break_off_sigma*sigma+0.5)
    filtersize = filtersize.astype(np.int)

    # 扩展边
    f = np.pad(f, ((filtersize,filtersize), (filtersize,filtersize)), 'edge')
    x = np.arange(-filtersize, filtersize+1)

    # 翻转滤波核
    x = x*-1
    Gauss = 1/(np.power(2*np.pi, 0.5) * sigma)* np.exp((x**2)/(-2 * sigma * sigma))

    print(Gauss)
    if iorder == 0:
        # 高斯滤波
        Gx = Gauss / sum(Gauss)
    elif iorder == 1:
        # 一阶求导
        Gx = -(x/sigma**2)*Gauss
        Gx = Gx/(np.sum(x*Gx))
    elif iorder == 2:
        # 二阶求导
        Gx = (x**2/sigma**4-1/sigma**2)*Gauss
        Gx = Gx-sum(Gx)/(2*filtersize+1)
        Gx = Gx/sum(0.5*x*x*Gx)

    # 扩展到二维
    Gx = Gx.reshape(1, -1)
    # Gx=np.transpose([Gx])

    # 卷积
    h = signal.convolve(f, Gx, mode="same")

    print(Gauss)
    if jorder == 0:
        Gy = Gauss / sum(Gauss)
    elif jorder == 1:
        Gy = -(x/sigma**2)*Gauss
        Gy = Gy/(np.sum(x*Gy))
    elif jorder == 2:
        Gy = (x**2/sigma**4-1/sigma**2)*Gauss
        Gy = Gy-sum(Gy)/(2*filtersize+1)
        Gy = Gy/sum(0.5*x*x*Gy)

    Gy = Gy.reshape(1,-1).T

    res=signal.convolve(h, Gy, mode="same")

    res2 = res[1:2, 1:2]
    end_h = (filtersize + H)
    end_w = (filtersize + W)
    res2 = np.array(res)[filtersize:end_h, filtersize:end_w]
    return res2


def NormDerivative(img, sigma, order):
    # 一阶求导
    if (order == 1):
        Ix = gDer(img, sigma, 1, 0)
        Iy = gDer(img, sigma, 0, 1)
        Iw = np.power(Ix**2 + Iy**2, 0.5)

    # 二阶求导
    if (order == 2):  # computes frobius norm
        Ix = gDer(img, sigma, 2, 0)
        Iy = gDer(img, sigma, 0, 2)
        Ixy = gDer(img, sigma, 1, 1)
        Iw = np.power(Ix**2 + Iy**2 + 4*Ixy, 0.5)
    return Iw


# njet是否edge  mink_norm shade的参数  sigma滤波和求导参数
def grey_edge(R, G, B, njet=0, mink_norm=1, sigma=1, saturation_threshold=255):
    """
    Estimates the light source of an input_image as proposed in:
    J. van de Weijer, Th. Gevers, A. Gijsenij
    "Edge-Based Color Constancy"
    IEEE Trans. Image Processing, accepted 2007.
    Depending on the parameters the estimation is equal to Grey-World, Max-RGB, general Grey-World,
    Shades-of-Grey or Grey-Edge algorithm.
    """
    mask_im = np.zeros(R.shape)
    img_max = np.maximum(R,G)
    img_max = np.maximum(img_max,B)

    # 移除所有饱和像素
    itemindex = np.where(img_max>= saturation_threshold)
    saturation_map = np.zeros(R.shape)
    saturation_map[itemindex] = 1

    # 扩散
    mask_im = dilation33(saturation_map)
    mask_im = 1 - mask_im

    # 移除边的像素生成最终的有效像素mask
    mask_im2 = set_border(mask_im, sigma+1, 0)
    # 不去掉饱和像素尤其是buiding图片差别很大
    # mask_im2 = np.ones(R.shape)

    if njet == 0:
        if (sigma != 0):
            # 去噪
            gauss_image_R = filters.gaussian(R, sigma=sigma, multichannel=True)
            gauss_image_G = filters.gaussian(G, sigma=sigma, multichannel=True)
            gauss_image_B = filters.gaussian(B, sigma=sigma, multichannel=True)
        else:
            gauss_image_R = R
            gauss_image_G = G
            gauss_image_B = B

        deriv_image_R = gauss_image_R[:, :]
        deriv_image_G = gauss_image_G[:, :]
        deriv_image_B = gauss_image_B[:, :]
    else:
       deriv_image_R = NormDerivative(R, sigma, njet)
       deriv_image_G = NormDerivative(G, sigma, njet)
       deriv_image_B = NormDerivative(B, sigma, njet)

    # estimate illuminations
    if mink_norm == -1:  # mink_norm = inf
        estimating_func = lambda x: np.max(x*mask_im2.astype(np.int))
    else:
        estimating_func = lambda x: np.power(np.sum(np.power(x*mask_im2.astype(np.int), mink_norm)), 1 / mink_norm)

    RS = np.sum(np.power(deriv_image_R, mink_norm))
    GS = np.sum(np.power(deriv_image_G, mink_norm))
    BS = np.sum(np.power(deriv_image_B, mink_norm))
    illum_R = estimating_func(deriv_image_R)
    illum_G = estimating_func(deriv_image_G)
    illum_B = estimating_func(deriv_image_B)

    illum_max = np.maximum(illum_R, illum_G)
    illum_max = np.maximum(illum_max, illum_B)

    R_gain = illum_max / illum_R
    G_gain = illum_max / illum_G
    B_gain = illum_max / illum_B

    return  R_gain,G_gain,B_gain


def apply_raw(pattern, R, GR, GB, B, R_gain, G_gain, B_gain, max):
    R = np.minimum(R * R_gain, max)
    GR = np.minimum(GR * G_gain, max)
    GB = np.minimum(GB * G_gain, max)
    B = np.minimum(B * B_gain, max)

    result_image = raw.bayer_channel_integration(R, GR, GB, B, pattern)

    return result_image


def apply_rgb(R, G, B, R_gain, G_gain, B_gain):
    h, w = R.shape

    img = np.zeros(shape=(h, w, 3))
    img[:,:,0] = np.minimum(R * R_gain, 255)
    img[:,:,1] = np.minimum(G * G_gain, 255)
    img[:,:,2] = np.minimum(B * B_gain, 255)

    return img


# 为了兼容RAW
def rgb_separation(image):
    image = image.astype(np.float)
    R=image[:,:,0]
    G=image[:,:,1]
    B=image[:,:,2]
    return R, G, B


# 为了兼容RGB的图片特殊的分割
def raw_awb_separation(image, pattern):
    image = image.astype(np.float)
    R, GR, GB, B = raw.bayer_channel_separation(image, pattern)
    G = (GR+GB)/2
    return R, GR, GB, B, G


def raw_white_balance():
    pattern = "GRBG"
    file_name = "D65_4032_2752_GRBG_1_LSC.raw"
    image = plained_raw.read_plained_file(file_name, 2752, 4032, 0)
    raw_image_show.raw_image_show_fullsize(image/1023, height=2752, width=4032)

    max = 1023
    type = "grey_edge"

    R, GR, GB, B, G = raw_awb_separation(image, pattern)
    if (type == "grey_world"):
        R_gain, G_gain, B_gain = grey_world(R, G, B)
    elif(type == "auto_threshold"):
        R_gain, G_gain, B_gain = auto_threshold(R, G, B)
    elif (type == "grey_world2"):
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=0, mink_norm=1, sigma=0, saturation_threshold=max)
    elif (type == "shade of grey"):
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=0, mink_norm=5, sigma=0, saturation_threshold=max)
    elif (type == "max_RGB"):
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=0, mink_norm=-1, sigma=0, saturation_threshold=max)
    elif (type == "grey_edge"):
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=1, mink_norm=5, sigma=2, saturation_threshold=max)
    print("grey_world R_gain, G_gain, B_gain=", R_gain, G_gain, B_gain)
    R_gain = R_gain * 1.0
    G_gain = G_gain * 1.0
    B_gain = B_gain * 1.0

    result_image = apply_raw(pattern, R, GR, GB, B, R_gain, G_gain, B_gain, max)

    raw_image_show.raw_image_show_fullsize(result_image/1023, height=2742, width=4032)
    raw_image_show.raw_image_show_fakecolor(result_image/1023, height=2752, width=4032, pattern="GRBG")

    h, w = R.shape
    img = np.zeros(shape=(h,w,3))
    img2 = np.zeros(shape=(h, w, 3))
    img[:,:,0] = R
    img[:,:,1] = G
    img[:,:,2] = B
    R2, GR2, GB2, B2 = raw.bayer_channel_separation(result_image, pattern)
    img2[:,:,0] = R2
    img2[:,:,1] = (GR2+GB2)/2
    img2[:,:,2] = B2
    color.rgb_show(img / 1023)
    color.rgb_show(img2 / 1023)
    print("[DebugMK]")


def RGB_white_balance():
    # image = plt.imread('building.jpg')
    image = cv2.imread("dianjing_1.jpg")
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if(np.max(image) <= 1):
        image = image * 255

    type = "grey_world"
    result_image = None

    R, G, B = rgb_separation(image)
    if (type == "grey_world"):
        R_gain, G_gain, B_gain = grey_world(R, G, B)
    elif(type == "auto_threshold"):
        R_gain, G_gain, B_gain = auto_threshold(R, G, B)
    elif (type == "grey_world2"):
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=0, mink_norm=1, sigma=0)
    elif (type == "shade_of_grey"):
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=0, mink_norm=5, sigma=0)
    elif (type == "max_RGB"):
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=0, mink_norm=-1, sigma=0)
    elif (type == "grey_edge"):
        R_gain, G_gain, B_gain = grey_edge(R, G, B, njet=1, mink_norm=5, sigma=2)
    print("grey_world R_gain, G_gain, B_gain=", R_gain, G_gain, B_gain)
    R_gain = R_gain * 1.1
    G_gain = G_gain * 1.0
    B_gain = B_gain * 1.0

    result_image = apply_rgb(R, G, B, R_gain, G_gain, B_gain)

    # color.rgb_show(image / 255)
    # color.rgb_show(result_image/255)
    cv2.imwrite("./result/grey_world_1_1.bmp", result_image)

    result_image = np.clip(result_image, 0, 255).astype(np.uint8)
    cv2.namedWindow('result_image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow("result_image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # RGB_white_balance()
    raw_white_balance()

    # img=1
    # sigma=2
    # gDer(img, sigma, 2, 2)
