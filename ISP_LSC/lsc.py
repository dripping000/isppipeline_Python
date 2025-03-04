import numpy as np
from matplotlib import pyplot as plt

import cv2

import plained_raw
import raw_image
import raw_image_show

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__)) + "\\"
rootpath = str(curPath)+"..\\"
syspath = sys.path
depth = rootpath.count("\\") - 1
sys.path = []
sys.path.append(rootpath)  # 将工程根目录加入到python搜索路径中
# sys.path.extend([rootpath+i for i in os.listdir(rootpath) if i[depth]!="."])  # 将工程目录下的一级目录添加到python搜索路径中
sys.path.extend(syspath)
# print(sys.path)
import ISP_BLC.blc


def apply_shading_to_image(img, block_size, shading_R, shading_GR, shading_GB, shading_B, pattern):
    R, GR, GB, B = raw_image.bayer_channel_separation(img, pattern)
    HH, HW = R.shape
    # 需要注意如果size不是整除的需要调整
    size_new = (HW + block_size, HH + block_size)
    # 插值
    ex_R_gain_map = cv2.resize(shading_R, size_new)
    ex_GR_gain_map = cv2.resize(shading_GR, size_new)
    ex_GB_gain_map = cv2.resize(shading_GB, size_new)
    ex_B_gain_map = cv2.resize(shading_B, size_new)
    half_b_size = int(block_size / 2)
    # 裁剪到原图大小
    R_gain_map = ex_R_gain_map[half_b_size:half_b_size + HH, half_b_size:half_b_size + HW]
    GR_gain_map = ex_GR_gain_map[half_b_size:half_b_size + HH, half_b_size:half_b_size + HW]
    GB_gain_map = ex_GB_gain_map[half_b_size:half_b_size + HH, half_b_size:half_b_size + HW]
    B_gain_map = ex_B_gain_map[half_b_size:half_b_size + HH, half_b_size:half_b_size + HW]
    # apply gain
    R_new = R * R_gain_map
    GR_new = GR * GR_gain_map
    GB_new = GB * GB_gain_map
    B_new = B * B_gain_map
    new_image = raw_image.bayer_channel_integration(R_new, GR_new, GB_new, B_new, "GRBG")

    new_image=np.clip(new_image,a_min=0,a_max=65535)  # [DebugMK]
    return new_image


def apply_shading_to_image_ratio(img, block_size_height, block_size_width, shading_R, shading_GR, shading_GB, shading_B, pattern, ratio, clip_range=[0, 2**10-1]):
    # 用G做luma
    luma_shading = (shading_GR + shading_GB) / 2
    # 计算调整之后luma shading
    new_luma_shading = luma_shading*ratio

    # 计算color shading
    R_color_shading = shading_R / luma_shading
    GR_color_shading = shading_GR / luma_shading
    GB_color_shading = shading_GB / luma_shading
    B_color_shading = shading_B / luma_shading

    # 合并两种shading
    new_shading_R = R_color_shading*new_luma_shading
    new_shading_GR = GR_color_shading*new_luma_shading
    new_shading_GB = GB_color_shading*new_luma_shading
    new_shading_B = B_color_shading*new_luma_shading

    R, GR, GB, B = raw_image.bayer_channel_separation(img, pattern)

    HH, HW = R.shape
    size_new = (HW + block_size_width, HH + block_size_height)

    # 插值的方法的选择
    ex_R_gain_map = cv2.resize(new_shading_R, size_new,interpolation=cv2.INTER_CUBIC)
    ex_GR_gain_map = cv2.resize(new_shading_GR, size_new,interpolation=cv2.INTER_CUBIC)
    ex_GB_gain_map = cv2.resize(new_shading_GB, size_new,interpolation=cv2.INTER_CUBIC)
    ex_B_gain_map = cv2.resize(new_shading_B, size_new,interpolation=cv2.INTER_CUBIC)

    # 裁剪到原图大小
    half_h_size = int(block_size_height / 2)
    half_w_size = int(block_size_width / 2)
    R_gain_map = ex_R_gain_map[half_h_size:half_h_size + HH, half_w_size:half_w_size + HW]
    GR_gain_map = ex_GR_gain_map[half_h_size:half_h_size + HH, half_w_size:half_w_size + HW]
    GB_gain_map = ex_GB_gain_map[half_h_size:half_h_size + HH, half_w_size:half_w_size + HW]
    B_gain_map = ex_B_gain_map[half_h_size:half_h_size + HH, half_w_size:half_w_size + HW]

    R_new = R * R_gain_map
    GR_new = GR * GR_gain_map
    GB_new = GB * GB_gain_map
    B_new = B * B_gain_map

    new_image = raw_image.bayer_channel_integration(R_new, GR_new, GB_new, B_new, pattern)
    # 值缩减到0~1023
    new_image = np.clip(new_image, a_min=clip_range[0], a_max=clip_range[1])
    return new_image


def create_lsc_data(img, block_size_height, block_size_width, pattern):
    # 分开四个颜色通道
    R, GR, GB, B = raw_image.bayer_channel_separation(img, pattern)

    # 每张的高宽
    HH, HW = R.shape
    # 生成分多少块
    Hblocks = int(HH/block_size_height)  # [DebugMK]
    Wblocks = int(HW/block_size_width)  # [DebugMK]
    print("Hblocks=%d, Wblocks=%d" % (Hblocks, Wblocks))
    # 结果预分配
    R_LSC_data = np.zeros((Hblocks, Wblocks))
    B_LSC_data = np.zeros((Hblocks, Wblocks))
    GR_LSC_data = np.zeros((Hblocks, Wblocks))
    GB_LSC_data = np.zeros((Hblocks, Wblocks))
    # 块距离中心的距离
    RA = np.zeros((Hblocks, Wblocks))

    # 中心点
    center_y = HH/2
    center_x = HW/2

    for y in range(0, HH, block_size_height):
        for x in range(0, HW, block_size_width):
            xx = x + block_size_width/2
            yy = y + block_size_height/2
            block_y_num = int(y / block_size_height)
            block_x_num = int(x / block_size_width)

            if ((x + block_size_width) > HW) or ((y + block_size_height) > HH):
                print("block_y_num=%d, block_x_num=%d, x: %d-%d, y: %d-%d" % (block_y_num, block_x_num, x, x + block_size_width, y, y + block_size_height))
            else:
                # 图像中心是光心
                RA[block_y_num, block_x_num] = (yy - center_y) * (yy - center_y) + (xx - center_x) * (xx - center_x)
                R_LSC_data[block_y_num, block_x_num] = R[y:y+block_size_height, x:x+block_size_width].mean()
                GR_LSC_data[block_y_num, block_x_num] = GR[y:y+block_size_height, x:x+block_size_width].mean()
                GB_LSC_data[block_y_num, block_x_num] = GB[y:y+block_size_height, x:x+block_size_width].mean()
                B_LSC_data[block_y_num, block_x_num] = B[y:y+block_size_height, x:x+block_size_width].mean()

    # 寻找光心块
    center_point = np.where(GR_LSC_data==np.max(GR_LSC_data))
    center_y = center_point[0]*block_size_height + block_size_height/2
    center_x = center_point[1]*block_size_width + block_size_width/2
    for y in range(0, HH, block_size_height):
        for x in range(0, HW, block_size_width):
            xx = x + block_size_width/2
            yy = y + block_size_height/2
            block_y_num=int(y / block_size_height)
            block_x_num =int(x / block_size_width)

            if ((x + block_size_width) > HW) or ((y + block_size_height) > HH):
                print("block_y_num=%d, block_x_num=%d, x: %d-%d, y: %d-%d" % (block_y_num, block_x_num, x, x + block_size_width, y, y + block_size_height))
            else:
                RA[block_y_num,block_x_num] = (yy - center_y) * (yy - center_y) + (xx - center_x) * (xx - center_x)

    raw_image_show.raw_image_show_3D(R_LSC_data,Hblocks,Wblocks)  # DebugMK
    plained_raw.write_plained_file("./Resource/new.raw",R_LSC_data)


    # 4个颜色数据通道展平
    RA_flatten = RA.flatten()
    R_LSC_data_flatten = R_LSC_data.flatten()
    GR_LSC_data_flatten = GR_LSC_data.flatten()
    GB_LSC_data_flatten = GB_LSC_data.flatten()
    B_LSC_data_flatten = B_LSC_data.flatten()

    # 最亮块的值
    Max_R=np.max(R_LSC_data_flatten)
    Max_GR=np.max(GR_LSC_data_flatten)
    Max_GB=np.max(GB_LSC_data_flatten)
    Max_B=np.max(B_LSC_data_flatten)

    # 得到gain，还没有外插
    G_R_LSC_data = Max_R/R_LSC_data
    G_GR_LSC_data = Max_GR/GR_LSC_data
    G_GB_LSC_data = Max_GB/GB_LSC_data
    G_B_LSC_data = Max_B/B_LSC_data
    # 暗角不补偿
    G_R_LSC_data[G_R_LSC_data > 10] = 1
    G_GR_LSC_data[G_GR_LSC_data > 10] = 1
    G_GB_LSC_data[G_GB_LSC_data > 10] = 1
    G_B_LSC_data[G_B_LSC_data > 10] = 1

    # show_1
    R_R = R_LSC_data_flatten/np.max(R_LSC_data_flatten)
    R_GR = GR_LSC_data_flatten/np.max(GR_LSC_data_flatten)
    R_GB = GB_LSC_data_flatten/np.max(GB_LSC_data_flatten)
    R_B = B_LSC_data_flatten/np.max(B_LSC_data_flatten)

    G_R = 1/R_R
    G_GR = 1/R_GR
    G_GB = 1/R_GB
    G_B = 1/R_B

    plt.figure()
    plt.scatter(RA_flatten, G_B, s=1, color='blue')
    plt.scatter(RA_flatten, G_GR, s=1, color='green')
    plt.scatter(RA_flatten, G_GB, s=1, color='green')
    plt.scatter(RA_flatten, G_R, s=1, color='red')
    plt.show()


    # 重要的拟合
    par_R = np.polyfit(RA_flatten, G_R, 3)
    par_GR = np.polyfit(RA_flatten, G_GR, 3)
    par_GB = np.polyfit(RA_flatten, G_GB, 3)
    par_B = np.polyfit(RA_flatten, G_B, 3)
    # 拟合之后生成所有点的值
    ES_R = par_R[0] * (RA_flatten**3) + par_R[1] * (RA_flatten**2) + par_R[2]* (RA_flatten ) + par_R[3]
    ES_GR = par_GR[0] * (RA_flatten**3) + par_GR[1] * (RA_flatten**2) + par_GR[2]* (RA_flatten ) + par_GR[3]
    ES_GB = par_GB[0] * (RA_flatten**3) + par_GB[1] * (RA_flatten**2) + par_GB[2]* (RA_flatten ) + par_GB[3]
    ES_B = par_B[0] * (RA_flatten**3) + par_B[1] * (RA_flatten**2) + par_B[2]* (RA_flatten ) + par_B[3]
    # show_2  # 拟合数据和原有数据有什么不同
    plt.figure()
    plt.scatter(RA_flatten, ES_B, s=1, color='blue')
    plt.scatter(RA_flatten, ES_GR, s=1, color='green')
    plt.scatter(RA_flatten, ES_GB, s=1, color='green')
    plt.scatter(RA_flatten, ES_R, s=1, color='red')
    plt.show()

    # 通过拟合的函数生成一个补偿gain的表
    EX_RA = np.zeros((Hblocks+2, Wblocks+2))
    EX_R = np.zeros((Hblocks+2, Wblocks+2))
    EX_GR = np.zeros((Hblocks+2, Wblocks+2))
    EX_GB= np.zeros((Hblocks+2, Wblocks+2))
    EX_B = np.zeros((Hblocks+2, Wblocks+2))

    new_center_y = center_point[0]+1
    new_center_x = center_point[1]+1
    for y in range(0, Hblocks+2):
         for x in range(0, Wblocks+2):
            EX_RA[y, x]=(y-new_center_y)*block_size_height*(y-new_center_y)*block_size_height+(x-new_center_x)*block_size_width*(x-new_center_x)*block_size_width

            EX_R[y, x]= par_R[0] * (EX_RA[y,x]**3)+par_R[1] * (EX_RA[y,x]**2) + par_R[2]* (EX_RA[y,x] )+par_R[3]
            EX_GR[y, x]= par_GR[0] * (EX_RA[y,x]**3)+par_GR[1] * (EX_RA[y,x]**2) + par_GR[2]* (EX_RA[y,x] )+par_GR[3]
            EX_GB[y, x]= par_GB[0] * (EX_RA[y,x]**3)+par_GB[1] * (EX_RA[y,x]**2) + par_GB[2]* (EX_RA[y,x] )+par_GB[3]
            EX_B[y, x]= par_B[0] * (EX_RA[y,x]**3)+par_B[1] * (EX_RA[y,x]**2) + par_B[2]* (EX_RA[y,x] )+par_B[3]

    # 中心用实际采样的数据
    EX_R[1:1+Hblocks, 1:1+Wblocks] = G_R_LSC_data
    EX_GR[1:1+Hblocks, 1:1+Wblocks] = G_GR_LSC_data
    EX_GB[1:1+Hblocks, 1:1+Wblocks] = G_GB_LSC_data
    EX_B[1:1+Hblocks, 1:1+Wblocks] = G_B_LSC_data

    raw_image_show.raw_image_show_3D(EX_R,Hblocks+2,Wblocks+2)  # DebugMK
    EX_R_debug = EX_R*1000
    plained_raw.write_plained_file("./Resource/new.raw",EX_R_debug)

    # return EX_R, EX_GR, EX_GB, EX_B  # mesh边缘拟合
    return G_R_LSC_data, G_GR_LSC_data, G_GB_LSC_data, G_B_LSC_data  # mesh


def interface(raw, width, height, BayerPatternType, clip_range):
    block_size = 16

    raw_lsc = plained_raw.read_plained_file("./Resource/LSC.raw", height, width, shift_bits=0)
    shading_R, shading_GR, shading_GB, shading_B = create_lsc_data(raw_lsc, block_size, BayerPatternType)

    raw_ = apply_shading_to_image_ratio(img=raw, block_size=block_size, shading_R=shading_R, shading_GR=shading_GR, shading_GB=shading_GB, shading_B=shading_B, pattern=BayerPatternType, ratio=1, clip_range=clip_range)

    return raw_


if __name__ == "__main__":
    width = 4096
    height = 3072

    pattern = "BGGR"
    clip_range = [0, 2**14-1]

    # block_size_height = int(height/2/13)
    # block_size_width = int(width/2/17)
    block_size_height = int(16)
    block_size_width = int(16)

    raw = plained_raw.read_plained_file("./Resource/20250303/p[MfsrBlend]_req[1]_batch[0]_BPS[1]_[in]_port[0]_w[4096]_h[3072]_stride[8192]_scanline[3072]_20250228_205435_564978.RawPlain16LSB1", height, width, shift_bits=0)

    raw = ISP_BLC.blc.interface(raw, width, height, pattern, clip_range)
    plained_raw.write_plained_file("./Resource/new_blc.raw",raw)

    shading_R, shading_GR, shading_GB, shading_B = create_lsc_data(raw, block_size_height, block_size_width, pattern)


    img2 = plained_raw.read_plained_file("./Resource/20250303/p[MfsrBlend]_req[1]_batch[0]_BPS[1]_[in]_port[0]_w[4096]_h[3072]_stride[8192]_scanline[3072]_20250228_205435_564978.RawPlain16LSB1", height, width, shift_bits=0)
    raw_image_show.raw_image_show_fullsize(img2/clip_range[1], height, width)

    # 普通的
    #apply_shading_to_image(img=img, block_size=block_size, shading_R=EX_R, shading_GR=EX_GR, shading_GB=EX_GB, shading_B=EX_B, pattern="GRBG")
    # luma和color shading
    new_image = apply_shading_to_image_ratio(img=img2, block_size_height=block_size_height, block_size_width=block_size_width, shading_R=shading_R, shading_GR=shading_GR, shading_GB=shading_GB, shading_B=shading_B, pattern=pattern, ratio=1, clip_range=clip_range)
    print(np.min(new_image), np.max(new_image))
    # raw_image_show.raw_image_show_3D(new_image,height,width)  # DebugMK
    plained_raw.write_plained_file("./Resource/new.raw",new_image)

    raw_image_show.raw_image_show_fakecolor(new_image/clip_range[1], height, width, pattern="BGGR")
    raw_image_show.raw_image_show_fullsize(new_image/clip_range[1], height, width)
    print(np.min(new_image), np.max(new_image))

