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
import ISP_NR.hvs_denoise
import ISP_Demosaic.demosaic
import ISP_Sharpen.sharpen


if __name__ == "__main__":
    print("ISPPipeline_start")

    # /* read raw */
    width = 768
    height = 512
    shift_bits = 0

    BayerPatternType = "RGGB"
    clip_range = [0, 1023]


    # /* read raw */
    raw = plained_raw.read_plained_file("./Resource/DSC16_1339_768x512_rggb_blc.raw", height, width, shift_bits)
    raw = raw.astype(np.float)


    # /* BNR */
    # raw = ISP_NR.hvs_denoise.interface(raw, width, height, BayerPatternType, clip_range)
    # plained_raw.DebugMK_raw("./Resource/test_BNR.bin", "./Resource/test_BNR.bmp", raw, clip_range)


    # /* Demosaic */
    image = ISP_Demosaic.demosaic.interface(raw, width, height, BayerPatternType, clip_range)
    plained_raw.DebugMK_raw("./Resource/test_Demosaic.bin", "./Resource/test_Demosaic.bmp", image, clip_range)


    # /* Sharpen */
    image = ISP_Sharpen.sharpen.interface(image, width, height, clip_range)
    plained_raw.DebugMK_raw("./Resource/test_Sharpen.bin", "./Resource/test_Sharpen.bmp", image, clip_range)


    print("ISPPipeline_end")
