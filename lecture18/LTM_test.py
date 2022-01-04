import color_utils as color
import numpy as np
import cv2 as cv
import math


def LTM(RGB,maxvalue=2**14,new_scale=30):
    height,width,C=RGB.shape
    d = 120# kernel size
    sigmaColor = 0.8 #color domain sigma
    sigmaSpace = 100#((width**2+height**2)**0.5)*0.02  # space domain sigma
    y =20*RGB[:,:,0]/61+40*RGB[:,:,1]/61+1*RGB[:,:,2]/61
    # ycc=color.rgb2ycbcr(RGB,width,height)
    # y = ycc[:, :, 0]
    # cb = ycc[:, :, 1]
    # cr = ycc[:, :, 2]
    chroma=np.empty_like(RGB)
    chroma[:,:,0]=RGB[:,:,0]/y
    chroma[:, :, 1] = RGB[:, :, 1] / y
    chroma[:, :, 2] = RGB[:, :, 2] / y
    log_y=np.log10(y)
    print(np.max(np.max(y)), np.min(np.min(y)))
    print(np.max(np.max(log_y)), np.min(np.min(log_y)))
    base = cv.bilateralFilter(log_y.astype(np.float32),d,sigmaColor,sigmaSpace)
    log_base = base
    print(np.max(np.max(log_base)), np.min(np.min(log_base)))
    detail_log_lum = log_y-log_base
    new_base=np.log10(new_scale)
    min_log_base=np.maximum(np.min(log_base),1)#为了放缩比准确
    new_base2=(np.max(log_base)-min_log_base)
    base_scale = new_base / new_base2
    large_scale2_reduced = log_base * base_scale
    log_absolute_scale = np.max(log_base)*base_scale
    print(np.max(log_base), np.min(log_base))
    print("xxx",new_base,new_base2,base_scale)
    out_log_lum = detail_log_lum + large_scale2_reduced - log_absolute_scale
    out_log_lum2=10**out_log_lum

    # print(np.max(detail_log_lum), np.min(detail_log_lum))
    # print(np.max(large_scale2_reduced), np.min(large_scale2_reduced))
    # print(np.max(out_log_lum), np.min(out_log_lum))
    # print(np.max(out_log_lum2),np.min(out_log_lum2))
    #out_log_lum2 = np.clip(out_log_lum2, 0, 1)
    chroma[:, :, 0] = chroma[:, :, 0] *out_log_lum2
    chroma[:, :, 1] = chroma[:, :, 1] *out_log_lum2
    chroma[:, :, 2] = chroma[:, :, 2] *out_log_lum2
    chroma=chroma-np.min(chroma)
    chroma=np.clip(chroma,0,1)
    #不做的话,色彩会偏色.
    chroma=chroma**(1/2.2)

    return chroma

if __name__ == "__main__":
    # read image
    maxvalue = 2**14-1
    image_BRG = cv.imread('9C4A0034-a460e29cd9.exr',cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    #image_BRG = cv.imread('smallOffice.hdr', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    image=np.empty_like(image_BRG)
    image[:,:,0]=image_BRG[:,:,2]
    image[:, :, 1] = image_BRG[:, :, 1]
    image[:, :, 2] = image_BRG[:, :, 0]
    print("max",np.max(image), np.min(image))

    image=image/np.max(image)
    if (np.max(image) <= 1):
        image = image * (maxvalue-1)
        print("max", np.max(image), np.min(image))
        image = np.round(image)
        image += 1
        print("max", np.max(image), np.min(image))
    new_image=LTM(image)
    print(np.max(image), np.min(image))
    print(np.max(new_image), np.min(new_image))
    color.rgb_show(image/np.max(image))
    color.rgb_show(new_image)
