import raw_image as raw
import raw_image_show
import read_plained_raw as plained_raw
import numpy as np
import matplotlib.pyplot as plt
import color_utils as color
from skimage import filters
from scipy import signal

#Gx = Gx.reshape(1, -1)
# Gx=np.transpose([Gx])
# 卷积
#h = signal.convolve(f, Gx, mode="same")
file_name = "kodim19_small.raw"
image = plained_raw.read_plained_file(file_name, 64, 64, 0)
raw_image_show.raw_image_show_fullsize(image / 1023, height=64, width=64)
kernel_V1=np.array([1, -1, 0])
kernel_V1=kernel_V1.reshape(1, -1)
b=signal.convolve(image, kernel_V1, 'same')
print(b)