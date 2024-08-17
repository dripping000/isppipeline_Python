import numpy as np
import json

import plained_raw
import raw_image
import raw_image_show

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
from isp_utils import plained_raw


def create_awb_stats_data(img, stats_size_width, stats_size_height, pattern):
    # 分开四个颜色通道
    R, GR, GB, B = raw_image.bayer_channel_separation(img, pattern)

    # 每张的高宽
    HH, HW = R.shape
    # 块尺寸
    block_size_height = int(HH/stats_size_height)
    block_size_width = int(HW/stats_size_width)
    # 结果预分配
    R_awb_stats = np.zeros((stats_size_height, stats_size_width))
    G_awb_stats = np.zeros((stats_size_height, stats_size_width))
    B_awb_stats = np.zeros((stats_size_height, stats_size_width))

    for y in range(0, HH, block_size_height):
        for x in range(0, HW, block_size_width):
            block_y_num = int(y / block_size_height)
            block_x_num = int(x / block_size_width)

            R_ = R[y:y+block_size_height, x:x+block_size_width].mean()
            GR_ = GR[y:y+block_size_height, x:x+block_size_width].mean()
            GB_ = GB[y:y+block_size_height, x:x+block_size_width].mean()
            G_ = (GR_+GB_)/2
            B_ = B[y:y+block_size_height, x:x+block_size_width].mean()

            # R_awb_stats[block_y_num, block_x_num] = R_ / G_
            # G_awb_stats[block_y_num, block_x_num] = G_ / G_
            # B_awb_stats[block_y_num, block_x_num] = B_ / G_

            R_awb_stats[block_y_num, block_x_num] = R_
            G_awb_stats[block_y_num, block_x_num] = G_
            B_awb_stats[block_y_num, block_x_num] = B_


    # raw_image_show.raw_image_show_3D(R_awb_stats,stats_size_height,stats_size_width)  # DebugMK
    # raw_image_show.raw_image_show_3D(G_awb_stats,stats_size_height,stats_size_width)  # DebugMK
    # raw_image_show.raw_image_show_3D(B_awb_stats,stats_size_height,stats_size_width)  # DebugMK

    fig =  plt.figure()
    ax = Axes3D(fig)
    ax = plt.subplot(1,1,1,projection='3d')
    X = np.arange(0, stats_size_width)
    Y = np.arange(0, stats_size_height)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    # ax.plot_surface(X, Y, R_awb_stats, rstride=1, cstride=1, cmap='rainbow')
    ax.plot_wireframe(X, Y, R_awb_stats, rstride=10, cstride=10, color='red')
    ax.plot_wireframe(X, Y, G_awb_stats, rstride=10, cstride=10, color='green')
    ax.plot_wireframe(X, Y, B_awb_stats, rstride=10, cstride=10, color='blue')
    plt.show()
    print('show')


    return R_awb_stats, G_awb_stats, B_awb_stats


def interface(raw, width, height, BayerPatternType, clip_range):

    pass


if __name__ == "__main__":
    width = 4096
    height = 3072

    pattern = "BGGR"
    clip_range = [0, 2**14-1]

    stats_size_width = 64
    stats_size_height = 48

    # 读取显示RAW文件
    raw = plained_raw.read_plained_file("./Resource/D65.raw", height, width, shift_bits=0)
    # raw_image_show.raw_image_show_fullsize(img/clip_range[1], height, width)

    raw = ISP_BLC.blc.interface(raw, width, height, pattern, clip_range)
    plained_raw.DebugMK_raw("./Resource/test_BLC.bin", "./Resource/test_BLC.bmp", raw, clip_range)

    # 生成awb stats
    R_awb_stats, G_awb_stats, B_awb_stats = create_awb_stats_data(raw, stats_size_width, stats_size_height, pattern)


    # 生成json文件
    # https://blog.csdn.net/weixin_44322778/article/details/131045935
    # 读取JSON文件
    with open('./Resource/isp_config_cannon.json', 'r') as f:
        data = json.load(f)

    # 打印JSON数据
    print(data)

    data['wb_gain']['R_awb_stats'] = R_awb_stats.flatten().tolist()
    data['wb_gain']['G_awb_stats'] = G_awb_stats.flatten().tolist()
    data['wb_gain']['B_awb_stats'] = B_awb_stats.flatten().tolist()

    # 保存修改后的JSON文件
    with open('./Resource/isp_config_cannon_.json', 'w') as f:
        json.dump(data, f, indent=4)

    pass
