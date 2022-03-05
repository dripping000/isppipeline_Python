import numpy as np
import random

import plained_raw
import raw_image_show


# SNR噪声的比例
def add_defect_pixel(img, SNR):
    img_ = img.copy()
    h, w = img_.shape

    mask = np.random.choice((0, 1, 2, 3, 4), size=(h, w), p=[SNR, (1 - SNR) / 4., (1 - SNR) / 4., (1 - SNR) / 4., (1 - SNR) / 4.])  # 随机的类别,大小,几率
    img_[mask == 1] = random.randint(1000,1024)  # 椒噪声
    img_[mask == 2] = random.randint(800,1000)  # 椒噪声
    img_[mask == 3] = random.randint(1,24)  # 椒噪声
    img_[mask == 4] = random.randint(25,200)  # 椒噪声

    return img_


if __name__ == "__main__":
    img = plained_raw.read_plained_file("DSC16_1339_768x512_rggb_blc.raw", height=512, width=768, shift_bits=0)

    img1 = add_defect_pixel(img, 0.998)
    print(np.max(img))
    plained_raw.write_plained_file("DSC16_1339_768x512_rggb_wait_dpc.raw", image=img1)

    raw_image_show.raw_image_show_fullsize(img1/(2**10), height=512, width=768)
