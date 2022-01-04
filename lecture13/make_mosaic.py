import raw_image as raw
import raw_image_show
import plained_raw as plained_raw
import numpy as np
import matplotlib.pyplot as plt
import color_utils as color
from skimage import filters
from scipy import signal

def make_mosaic(im,pattern):
    w,h,z=im.shape
    R=np.zeros((w,h))
    GR = np.zeros((w, h))
    GB = np.zeros((w, h))
    B = np.zeros((w, h))
    image_data=im
    #将对应位置的元素取出来,因为懒所以没有用效率最高的方法,大家可以自己去实现
    if (pattern == "RGGB"):
        R[::2, ::2]= image_data[::2, ::2, 0]
        GR[::2, 1::2] = image_data[::2, 1::2, 1]
        GB[1::2, ::2] = image_data[1::2, ::2, 1]
        B[1::2, 1::2]= image_data[1::2, 1::2, 2]
    elif (pattern == "GRBG"):
        GR[::2, ::2] = image_data[::2, ::2, 1]
        R[::2, 1::2] = image_data[::2, 1::2, 0]
        B[1::2, ::2] = image_data[1::2, ::2, 2]
        GB[1::2, 1::2] = image_data[1::2, 1::2, 1]
    elif (pattern == "GBRG"):
        GB[::2, ::2] = image_data[::2, ::2, 1]
        B[::2, 1::2] = image_data[::2, 1::2, 2]
        R[1::2, ::2] = image_data[1::2, ::2, 0]
        GR[1::2, 1::2] = image_data[1::2, 1::2, 1]
    elif (pattern == "BGGR"):
        B[::2, ::2] = image_data[::2, ::2, 2]
        GB[::2, 1::2] = image_data[::2, 1::2, 1]
        GR[1::2, ::2] = image_data[1::2, ::2, 1]
        R[1::2, 1::2] = image_data[1::2, 1::2, 0]
    else:
        print("pattern must be one of :  RGGB, GBRG, GBRG, or BGGR")
        return
    result_image=R+GR+GB+B
    return result_image

    # X(:, 1: 2:end, 1)=0;
    # X(2: 2:end, 2: 2:end, 1)=0;
    # X(2: 2:end, 1: 2:end, 2)=0;
    # X(1: 2:end, 2: 2:end, 2)=0;
    # X(1: 2:end, 1: 2:end, 3)=0;
    # X(:, 2: 2:end, 3)=0;
if __name__ == "__main__":
    #read_image
    maxvalue=255
    pattern="GRBG"
    image = plt.imread('kodim19.png')
    if(np.max(image)<=1):
        image =image*maxvalue
    result_image=make_mosaic(image,pattern)
    h,w=result_image.shape
    raw_image_show.raw_image_show_fakecolor(result_image/maxvalue,h,w,pattern)
    plained_raw.write_plained_file("kodim19.raw",result_image)
    small_result_image = result_image[384:384+64,256:256+64]
    raw_image_show.raw_image_show_fakecolor(small_result_image / maxvalue, 64, 64, pattern)
    plained_raw.write_plained_file("kodim19_small.raw", small_result_image)
    #raw_white_balance()