import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("lena_gray.bmp", 0)
img=img.astype(np.int16)
gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
#为了显示效果
res = cv2.convertScaleAbs(gray_lap)

plt.imshow(res, 'gray')
plt.title('Result Image')
plt.axis('off')
plt.show()
