import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__)) + "\\"
rootpath = str(curPath)
syspath = sys.path
depth = rootpath.count("\\") - 1
sys.path = []
sys.path.append(rootpath)  # 将工程根目录加入到python搜索路径中
sys.path.extend([rootpath+i for i in os.listdir(rootpath) if i[depth]!="."])  # 将工程目录下的一级目录添加到python搜索路径中
sys.path.extend(syspath)
# print(sys.path)

import numpy as np

from isp_utils import plained_raw
from isp_utils import raw_image_show

import ISP_BLC.blc
import ISP_DPC.dpc
import ISP_NR.hvs_denoise
import ISP_Demosaic.demosaic
import ISP_Sharpen.sharpen
import ISP_LTM.ltm


if __name__ == "__main__":
    print("ISPPipeline_start")


    BLC_flag = 0
    DPC_flag = 0
    BNR_flag = 0

    Demosaic_flag = 1
    Sharpen_flag = 1
    LTM_flag = 1


    # /* read raw */
    file_name = "./Resource/DSC16_1339_768x512_rggb_wait_dpc.raw"

    width = 768
    height = 512
    shift_bits = 0

    BayerPatternType = "RGGB"
    clip_range = [0, 2**10-1]


    # /* read raw */
    raw = plained_raw.read_plained_file(file_name, height, width, shift_bits)
    raw_image_show.raw_image_show_fullsize(raw/clip_range[1], height, width)
    print("/* read raw */ shape", raw.shape)
    print("/* read raw */ max", np.max(raw), "min", np.min(raw), "\n")
    raw = raw.astype(np.float)


    if BLC_flag == 1:
        # /* BLC */
        # clip_range--->clip_range
        raw = ISP_BLC.blc.interface(raw, width, height, BayerPatternType, clip_range)
        plained_raw.DebugMK_raw("./Resource/test_BLC.bin", "./Resource/test_BLC.bmp", raw, clip_range)


    if DPC_flag == 1:
        # /* DPC */
        # clip_range--->clip_range
        raw = ISP_DPC.dpc.interface(raw, width, height, BayerPatternType, clip_range)
        plained_raw.DebugMK_raw("./Resource/test_DPC.bin", "./Resource/test_DPC.bmp", raw, clip_range)


    if BNR_flag == 1:
        # /* BNR */
        # clip_range--->clip_range
        raw = ISP_NR.hvs_denoise.interface(raw, width, height, BayerPatternType, clip_range)
        plained_raw.DebugMK_raw("./Resource/test_BNR.bin", "./Resource/test_BNR.bmp", raw, clip_range)


    if Demosaic_flag == 1:
        # /* Demosaic */
        # clip_range--->clip_range gamma_off
        image = ISP_Demosaic.demosaic.interface(raw, width, height, BayerPatternType, clip_range)
        print("/* Demosaic */ shape", image.shape)
        print("/* Demosaic */ max", np.max(image), "min", np.min(image), "\n")
        plained_raw.DebugMK_raw("./Resource/test_Demosaic.bin", "./Resource/test_Demosaic.bmp", image, clip_range)


    if (Demosaic_flag == 1) and (Sharpen_flag == 1):
        # /* Sharpen */
        # clip_range--->clip_range
        image = ISP_Sharpen.sharpen.interface(image, width, height, clip_range)
        print("/* Sharpen */ shape", image.shape)
        print("/* Sharpen */ max", np.max(image), "min", np.min(image), "\n")
        plained_raw.DebugMK_raw("./Resource/test_Sharpen.bin", "./Resource/test_Sharpen.bmp", image, clip_range)


    if (Demosaic_flag == 1) and (LTM_flag == 1):
        # /* LTM */
        # clip_range--->clip_range gamma_on
        image = ISP_LTM.ltm.interface(image, width, height, clip_range)
        print("/* LTM */ shape", image.shape)
        print("/* LTM */ max", np.max(image), "min", np.min(image), "\n")
        plained_raw.DebugMK_raw("./Resource/test_LTM.bin", "./Resource/test_LTM.bmp", image, clip_range)


    print("ISPPipeline_end")
