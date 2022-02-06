import numpy as np
from matplotlib import pyplot as plt

import cv2 as cv

import color_utils as color


def LTM(RGB, maxvalue=(2**14-1), new_scale=30):
    height, width, C = RGB.shape

    d = 120  # kernel size
    sigmaColor = 0.8  # color domain sigma
    sigmaSpace = 100  # ((width**2+height**2)**0.5)*0.02  # space domain sigma

    y = 20*RGB[:,:,0]/61 + 40*RGB[:,:,1]/61 + 1*RGB[:,:,2]/61
    # ycc = color.rgb2ycbcr(RGB, width, height)
    # y = ycc[:, :, 0]
    # cb = ycc[:, :, 1]
    # cr = ycc[:, :, 2]

    chroma = np.empty_like(RGB)
    chroma[:, :, 0] = RGB[:, :, 0] / y
    chroma[:, :, 1] = RGB[:, :, 1] / y
    chroma[:, :, 2] = RGB[:, :, 2] / y

    log_y = np.log10(y)
    print("input_intensity max", np.max(y), "min", np.min(y))
    print("log(input_intensity) max", np.max(log_y), "min", np.min(log_y))
    print("\n")

    base = cv.bilateralFilter(log_y.astype(np.float32), d, sigmaColor, sigmaSpace)

    log_base = base
    print("base max", np.max(log_base), "min", np.min(log_base))

    detail_log_lum = log_y - log_base
    print("detail_log_lum max", np.max(detail_log_lum), "min", np.min(detail_log_lum))
    print("\n")

    # DebugMK
    new_base = np.log10(new_scale)
    min_log_base = np.maximum(np.min(log_base), 1)  # 为了放缩比准确
    new_base2 = (np.max(log_base) - min_log_base)
    base_scale = new_base / new_base2
    print("new_base=%lf, new_base2=%lf, base_scale=%lf" % (new_base, new_base2, base_scale))
    # print(new_base, new_base2, base_scale)
    print("\n")

    large_scale2_reduced = log_base * base_scale
    log_absolute_scale = np.max(log_base) * base_scale

    out_log_lum = large_scale2_reduced + detail_log_lum - log_absolute_scale
    out_log_lum2 = 10**out_log_lum

    print("log(base)*compressionfactor max", np.max(large_scale2_reduced), "min", np.min(large_scale2_reduced))
    print("log(output_intensity) max", np.max(out_log_lum), "min", np.min(out_log_lum))
    print("10^log(output_intensity) max", np.max(out_log_lum2), "min", np.min(out_log_lum2))
    print("\n")
    # out_log_lum2 = np.clip(out_log_lum2, 0, 1)

    chroma[:, :, 0] = chroma[:, :, 0] * out_log_lum2
    chroma[:, :, 1] = chroma[:, :, 1] * out_log_lum2
    chroma[:, :, 2] = chroma[:, :, 2] * out_log_lum2
    print("chroma max", np.max(chroma), "min", np.min(chroma))

    chroma = chroma - np.min(chroma)
    print("chroma max", np.max(chroma), "min", np.min(chroma))

    chroma = np.clip(chroma, 0, 1)  # DebugMK
    print("chroma max", np.max(chroma), "min", np.min(chroma))

    # gamma 不做的话,色彩会偏色.
    chroma = chroma**(1/2.2)

    return chroma


def interface(image, width, height, clip_range):
    maxvalue = clip_range[1]

    # image = image / np.max(image)  # 归一化
    image = image / maxvalue  # 归一化
    if (np.max(image) <= 1):
        image = image * maxvalue
        print("image_src*maxvalue max", np.max(image), "min", np.min(image))

        image = np.round(image)
        image += 1
        print("int(image_src*maxvalue)+1 max", np.max(image), "min", np.min(image))
        print("\n")

    new_image = LTM(image, maxvalue=maxvalue)
    print("image_result max", np.max(new_image), "min", np.min(new_image))

    new_image = new_image * maxvalue

    return new_image


if __name__ == "__main__":
    '''    '''
    x = np.arange(0, (2**14 - 1)+1, 1)  # np.arange(start, end+step, step)  [start, end] end/step+1

    y = np.log10(x)

    plt.plot(x, y, color='r')
    plt.show()


    # read image
    maxvalue = 2**14 - 1

    # image_BRG = cv.imread('9C4A0034-a460e29cd9.exr',cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    image_BRG = cv.imread('smallOffice.hdr', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)

    image = np.empty_like(image_BRG)
    image[:, :, 0] = image_BRG[:, :, 2]
    image[:, :, 1] = image_BRG[:, :, 1]
    image[:, :, 2] = image_BRG[:, :, 0]
    print("image_src max", np.max(image), "min", np.min(image))

    image = image / np.max(image)  # 归一化
    if (np.max(image) <= 1):
        image = image * maxvalue
        print("image_src*maxvalue max", np.max(image), "min", np.min(image))

        image = np.round(image)
        image += 1
        print("int(image_src*maxvalue)+1 max", np.max(image), "min", np.min(image))
        print("\n")

    new_image = LTM(image)
    print("image_result max", np.max(new_image), "min", np.min(new_image))

    color.rgb_show(image/np.max(image))
    color.rgb_show(new_image)
