import numpy as np
import cv2

import plained_raw
import raw_image
import raw_image_show


# 所有像素减去一个值
def simple_blc(img, blacklevel):
    img=img.astype(np.int16)

    img= img - blacklevel

    img=np.clip(img,a_min=0, a_max=np.max(img))
    return img


# 每个channel减去一个值
def blc_process(img, blacklevel1, blacklevel2, blacklevel3, blacklevel4):
    img = img.astype(np.int16)

    C1, C2, C3, C4 = raw_image.simple_separation(img)
    C1 = C1 - blacklevel1
    C2 = C2 - blacklevel2
    C3 = C3 - blacklevel3
    C4 = C4 - blacklevel4
    img = raw_image.simple_integration(C1, C2, C3, C4)

    img=np.clip(img, a_min=0, a_max=np.max(img))
    return img


# 用块的值进行插值
def block_blc(img, blacklevel1, blacklevel2, blacklevel3, blacklevel4):
    img = img.astype(np.int16)

    C1, C2, C3, C4 = raw_image.simple_separation(img)
    size = C1.shape
    size_new = (size[1], size[0])
    blacklevel1=cv2.resize(blacklevel1,size_new)
    blacklevel2=cv2.resize(blacklevel2,size_new)
    blacklevel3=cv2.resize(blacklevel3,size_new)
    blacklevel4=cv2.resize(blacklevel4,size_new)
    C1 = C1 - blacklevel1
    C2 = C2 - blacklevel2
    C3 = C3 - blacklevel3
    C4 = C4 - blacklevel4
    img = raw_image.simple_integration(C1, C2, C3, C4)

    img=np.clip(img,a_min=0,a_max=np.max(img))
    return img


if __name__ == "__main__":
    print('This is main of BLC')

    img = plained_raw.read_plained_file("DSC16_1339_768x512_rggb.raw", height=512, width=768, shift_bits=0)
    img1 = blc_process(img, 38, 38, 38, 38)

    raw_image_show.raw_image_show_fakecolor(img1, 512, 768, "BGGR")

    # blacklevel = np.ones((32, 48)) * 38
    # img1 = block_blc(img,blacklevel,blacklevel,blacklevel,blacklevel)

    plained_raw.write_plained_file("DSC16_1339_768x512_rggb_blc.raw", image=img1)
