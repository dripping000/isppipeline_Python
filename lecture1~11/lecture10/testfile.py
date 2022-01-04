
import numpy as np
import math, os, sys
from matplotlib import pyplot as plt
import raw_image_show
import raw_image
import BLC
import plained_raw
img=plained_raw.read_plained_file("D65_4076_2786_GRBG_1.raw",height=2786,width=4076,shift_bits=0)
raw_image_show.raw_image_show_fullsize(img / 1023,height=2752,width=4032)
img=BLC.simple_blc(img,64)
img=img[0:2752,0:4032]
raw_image_show.raw_image_show_fullsize(img / 1023,height=2752,width=4032)
plained_raw.write_plained_file("D65_4032_2752_GRBG_1_BLC.raw",image=img)

img=plained_raw.read_plained_file("D65_4076_2786_GRBG_2.raw",height=2786,width=4076,shift_bits=0)
img=BLC.simple_blc(img,64)
img=img[0:2752,0:4032]
raw_image_show.raw_image_show_fullsize(img / 1023,height=2752,width=4032)
plained_raw.write_plained_file("D65_4032_2752_GRBG_2_BLC.raw",image=img)


