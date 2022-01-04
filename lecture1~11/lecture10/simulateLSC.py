import numpy as np
import math

import raw_image_show


simulate_image = np.ones([600,800])
heigth,width = simulate_image.shape
center_x = width/2-0.5
center_y = heigth/2-0.5
# FOV
fov = 88
# 焦距
fl = (pow(pow(width/2,2)+pow(heigth/2,2),0.5))/math.tan(math.radians(fov/2))

for i in range(0,heigth):
    for j in range(0,width):
        distance=pow(pow((i-center_y), 2) + pow((j-center_x) , 2), 0.5)
        angle=math.atan(distance/fl)
        simulate_image[i,j]=simulate_image[i,j]*pow(math.cos(angle),4)

raw_image_show.raw_image_show_fullsize(simulate_image,heigth,width)
raw_image_show.raw_image_show_3D(simulate_image,heigth,width)

