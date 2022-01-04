import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('lena_gray.bmp', 0)
#16bit的有符号转换
img=img.astype(np.int16)
#用opencv是因为我们需要实部和虚部都进行处理
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
fshift = np.fft.fftshift(dft)

rows, cols = img.shape
crow,ccol = int(rows/2), int(cols/2) #中心位置
#opencv的实部,虚部是分别存储的.被滤掉的为0.
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
#显示频域滤波模板
plt.imshow(mask[:,:,1], 'gray')
plt.title('mask')
plt.axis('off')
plt.show()
#低频滤波
f = fshift * mask
print(f.shape, fshift.shape, mask.shape)
#高频滤波
f2 = fshift * (1-mask)
print(f2.shape, fshift.shape, mask.shape)
#去滤波后的幅度和相位
s1 = np.log(cv2.magnitude(fshift[:,:,0],fshift[:,:,1]))
ph_1 = cv2.phase(fshift[:,:,0],fshift[:,:,1])
s2 = np.log(cv2.magnitude(f[:,:,0],f[:,:,1]))
ph_2 = cv2.phase(f[:,:,0],f[:,:,1])
s3 = np.log(cv2.magnitude(f2[:,:,0],f2[:,:,1]))
ph_3 = cv2.phase(f2[:,:,0],f2[:,:,1])

plt.imshow(s1,'gray'),plt.title('original FFT'),plt.show()

plt.imshow(s2,'gray'),plt.title('center'),plt.show()

plt.imshow(s3,'gray'),plt.title('center'),plt.show()

plt.imshow(ph_1,'gray'),plt.title('original'),plt.show()

plt.imshow(ph_2,'gray'),plt.title('center'),plt.show()

plt.imshow(ph_3,'gray'),plt.title('center'),plt.show()

#把中心移到边角
ishift = np.fft.ifftshift(f)
iimg = cv2.idft(ishift)
#实际我们需要得到的图像是IDFT之后的幅度
res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])

ishift = np.fft.ifftshift(f2)
iimg = cv2.idft(ishift)
res2 = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])


plt.imshow(img, 'gray')
plt.title('Original Image')
plt.axis('off')
plt.show()
plt.imshow(res, 'gray')
plt.title('Low Image')
plt.axis('off')
plt.show()
plt.imshow(res2, 'gray')
plt.title('High Image')
plt.axis('off')
plt.show()
