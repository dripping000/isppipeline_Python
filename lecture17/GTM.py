import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splev, splrep

import cv2 as cv

import color_utils as color


def RGB_luminance(RGB):
    Y = 0.299*RGB[:,:,0]+0.5877*RGB[:,:,1]+0.114*RGB[:,:,2]
    return Y


def log_average(a, epsilon=sys.float_info.epsilon):
    a = a.astype(np.float)
    average = np.exp(np.average(np.log(a + epsilon)))
    return average


def tonemapping_operator_simple(RGB):
    RGB = RGB.astype(np.float)
    return RGB / (RGB + 1)


def tonemapping_operator_gamma(RGB, gamma=1, EV=0):
    RGB = RGB.astype(np.float)

    exposure = 2 ** EV
    RGB = (exposure * RGB) ** (1 / gamma)

    return RGB


def tonemapping_operator_logarithmic(RGB,
                                     q=1,
                                     k=1
                                     ):

    RGB = RGB.astype(np.float)

    q = 1 if q < 1 else q
    k = 1 if k < 1 else k

    L = RGB_luminance(RGB)
    L_max = np.max(L)
    L_d = np.log10(1 + L * q) / np.log10(1 + L_max * k)

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]
    #调整广播

    return RGB


def tonemapping_operator_normalisation(RGB):

    RGB = RGB.astype(np.float)

    L = RGB_luminance(RGB)
    L_max = np.max(L)

    RGB = RGB / L_max

    return RGB


def tonemapping_operator_exponential(RGB,
                                     q=1,
                                     k=1):

    RGB = RGB.astype(np.float)

    q = 1 if q < 1 else q
    k = 1 if k < 1 else k

    L = RGB_luminance(RGB)
    L_a = log_average(L)
    L_d = 1 - np.exp(-(L * q) / (L_a * k))

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    return RGB


def tonemapping_operator_logarithmic_mapping(
        RGB, p=1, q=1):

    RGB = RGB.astype(np.float)

    L = RGB_luminance(RGB)

    L_max = np.max(L)
    L_d = (np.log(1 + p * L) / np.log(1 + p * L_max)) ** (1 / q)

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    return RGB


def tonemapping_operator_exponentiation_mapping(
        RGB, p=1, q=1):

    RGB = RGB.astype(np.float)

    L = RGB_luminance(RGB)
    L_max = np.max(L)
    L_d = (L / L_max) ** (p / q)

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    return RGB


def tonemapping_operator_Schlick1994(RGB,
                                     p=1
                                     ):

    RGB = RGB.astype(np.float)

    L = RGB_luminance(RGB)
    L_max = np.max(L)
    L_d = (p * L) / (p * L - L + L_max)

    RGB = RGB * L_d[..., np.newaxis] / L[..., np.newaxis]

    return RGB


def tonemapping_operator_Tumblin1999(RGB,
                                     L_da=20,
                                     C_max=100,
                                     L_max=100
                                     ):

    RGB = RGB.astype(np.float)

    L_w = RGB_luminance(RGB)

    def f(x):
        return np.where(x > 100, 2.655,
                        1.855 + 0.4 * np.log10(x + 2.3 * 10 ** -5))

    L_wa = np.exp(np.mean(np.log(L_w + 2.3 * 10 ** -5)))
    g_d = f(L_da)
    g_w = f(L_wa)
    g_wd = g_w / (1.855 + 0.4 * np.log(L_da))

    mL_wa = np.sqrt(C_max) ** (g_wd - 1)

    L_d = mL_wa * L_da * (L_w / L_wa) ** (g_w / g_d)

    RGB = RGB * L_d[..., np.newaxis] / L_w[..., np.newaxis]
    RGB = RGB / L_max

    return RGB


def tonemapping_operator_Reinhard2004(RGB,
                                      f=0,
                                      m=0.3,
                                      a=0,
                                      c=0
                                     ):

    RGB = RGB.astype(np.float)

    C_av = np.array((np.average(RGB[..., 0]), np.average(RGB[..., 1]),
                     np.average(RGB[..., 2])))

    L = RGB_luminance(RGB)

    L_lav = log_average(L)
    L_min, L_max = np.min(L), np.max(L)

    f = np.exp(-f)

    m = (m if m > 0 else (0.3 + 0.7 * (
        (np.log(L_max) - L_lav) / (np.log(L_max) - np.log(L_min)) ** 1.4)))

    I_l = (c * RGB + (1 - c)) * L[..., np.newaxis]
    I_g = c * C_av + (1 - c) * L_lav
    I_a = a * I_l + (1 - a) * I_g

    RGB = RGB / (RGB + (f * I_a) ** m)

    return RGB


def tonemapping_operator_filmic(RGB,
                                shoulder_strength=0.22,
                                linear_strength=0.3,
                                linear_angle=0.1,
                                toe_strength=0.2,
                                toe_numerator=0.01,
                                toe_denominator=0.3,
                                exposure_bias=2,
                                linear_whitepoint=11.2):

    RGB = RGB.astype(np.float)

    A = shoulder_strength
    B = linear_strength
    C = linear_angle
    D = toe_strength
    E = toe_numerator
    F = toe_denominator

    def f(x, A, B, C, D, E, F):
        return ((
            (x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F)

    RGB = f(RGB * exposure_bias, A, B, C, D, E, F)
    RGB = RGB * (1 / f(linear_whitepoint, A, B, C, D, E, F))

    return RGB


def tonemapping_operator_aces(RGB,adapted_lum=1):

    RGB = RGB.astype(np.float)

    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14
    RGB =RGB
    RGB=(RGB * (A * RGB + B)) / (RGB * (C * RGB + D) + E)
    return RGB


# 下面都是曲线
def gamma_cr(x, gamma):
    y = x ** (gamma)
    return y


def logarithmic_cr(x, q=1, k=1):
    x = x.astype(np.float)

    q = 1 if q < 1 else q
    k = 1 if k < 1 else k

    L_max = np.max(x)
    L_d = np.log10(1+q*x)/np.log10(1+k*L_max)
    y= x*L_d/x
    return y


def sigmod_cr(x, b, c):
    x = x.astype(np.int64)  # 数据类型越界
    y = ((x**b)/(((c)**b)+(x**b)))
    return y


def log_normalizing(x, q=10, K=50):
    y = np.log10(1+q*x)/np.log10(1+K*255)
    return y


# 常见的用法
def global_tone_mapping(HDRIMG, s=0.6, gamma=2.2, max_value=1):
    HDRIMG = HDRIMG / np.max(HDRIMG)

    LDRIMG = np.empty_like(HDRIMG)

    X = np.empty((HDRIMG.shape[0], HDRIMG.shape[1]))
    LOG_X = np.empty_like(X)
    LOG_X_0 = np.empty(1)
    LOG_X_hat = np.empty_like(X)

    DBL_MIN = sys.float_info.min
    X_0 = np.max(HDRIMG)

    # Gamma compression
    X = HDRIMG

    LOG_X_0 = np.log2(X_0 + DBL_MIN)
    LOG_X = np.log2(X + DBL_MIN)
    LOG_X_hat = s * (LOG_X - LOG_X_0) + LOG_X_0

    # Restore log(X_hat) to X_hat, and store them to LDRIMG
    np.power(2.0, LOG_X_hat, LDRIMG)

    # Gamma Correction
    np.power(LDRIMG, (1.0/gamma), LDRIMG)

    # Fix out of range pixels
    LDRIMG[LDRIMG < 0.0] = 0.0
    LDRIMG[LDRIMG > 1.0] = 1.0
    LDRIMG = np.round(LDRIMG*255).astype("uint8")
    return LDRIMG


def apply_matrix(input_array, matrix):
    img_out = np.zeros_like(input_array)
    for c in (0, 1, 2):
        img_out[:, :, c] = matrix[c, 0] * input_array[:, :, 0] + \
                           matrix[c, 1] * input_array[:, :, 1] + \
                           matrix[c, 2] * input_array[:, :, 2]
    return img_out


RGB_TO_YCBCR = np.array([[0.299, 0.587, 0.144],
                         [-0.168736, -0.331264, 0.5],
                         [0.5, -0.418688, -0.081312]])

def tone_curve_correction(img_rgb, xs=(0, 64, 128, 192, 256), ys=(0, 64, 128, 192, 256)):
    func = splrep(xs, ys)
    img_ycbcr = apply_matrix(img_rgb * 256, RGB_TO_YCBCR)
    img_ycbcr[:, :, 0] = splev(img_ycbcr[:, :, 0], func)
    ycbcr2rgb = np.linalg.inv(RGB_TO_YCBCR)
    img_rgb_out = apply_matrix(img_ycbcr, ycbcr2rgb)
    return img_rgb_out / 256


if __name__ == "__main__":
    maxvalue = 1

    # read_image
    image_BRG = cv.imread('9C4A0034-a460e29cd9.exr', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    image = np.empty_like(image_BRG)
    image[:, :, 0] = image_BRG[:, :, 2]
    image[:, :, 1] = image_BRG[:, :, 1]
    image[:, :, 2] = image_BRG[:, :, 0]
    if (np.max(image) <= 1):
        image = image * maxvalue

    # new_image = global_tone_mapping(image)
    # new_image = gamma_cr(image, 1/2.2)
    new_image = image * 255  # DebugMK
    new_image = sigmod_cr(new_image, 0.7, 8)  # 2 16
    new_image = gamma_cr(new_image, 1/2.2)
    # new_image = log_normalizing(new_image, q=15, K=100)

    new_image2 = tone_curve_correction(new_image)
    # new_image2 = tonemapping_operator_filmic(new_image)

    color.rgb_show(image)
    color.rgb_show(new_image)
    color.rgb_show(new_image2)


    x = np.arange(256)
    # x = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])

    # y = sigmod_cr(x, 1, 100)
    # y1 = sigmod_cr(x, 2, 100)
    # y2 = sigmod_cr(x, 4, 100)
    # plt.plot(x, y, color='r')
    # plt.plot(x, y1, color='g')
    # plt.plot(x, y2, color='b')
    # plt.show()

    # y = log_normalizing(x, q=15, K=100)
    # y1 = logarithmic_cr(x, q=15, k=100)
    # y2 = np.log10(x)
    # plt.plot(x, y, color='r')
    # plt.plot(x, y1, color='g')
    # plt.plot(x, y2, color='b')
    # plt.show()
