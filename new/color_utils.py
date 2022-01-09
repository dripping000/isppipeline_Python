
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math



def mono_image_show(image, width, height,compress_ratio=1):
    x = width/(compress_ratio*100)
    y = height/(compress_ratio*100)
    plt.figure(num='test', figsize=(x, y))
    plt.imshow(image, cmap='gray', interpolation='bicubic', vmax=1.0)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()


# internal function
def labf(t):
    d = t**(1/3)
    index = np.where(t<=0.008856)
    d[index] = 7.787 * t[index] + 16 / 116
    return d


def RGB2LAB(X):
    a = np.array([
        [3.40479,  -1.537150, -0.498535],
        [-0.969256, 1.875992, 0.041556],
        [0.055648, -0.204043, 1.057311]])
    ai = np.linalg.inv(a)
    print(ai)

    h, w, c = X.shape

    R = X[:, :, 0]
    G = X[:, :, 1]
    B = X[:, :, 2]
    planed_R = R.flatten()
    planed_G = G.flatten()
    planed_B = B.flatten()

    planed_image = np.zeros((c, h*w))
    planed_image[0, :] = planed_R
    planed_image[1, :] = planed_G
    planed_image[2, :] = planed_B

    planed_lab = np.dot(ai, planed_image)
    planed_1 = planed_lab[0, :]
    planed_2 = planed_lab[1, :]
    planed_3 = planed_lab[2, :]
    L1 = np.reshape(planed_1, (h,w))
    L2 = np.reshape(planed_2, (h,w))
    L3 = np.reshape(planed_3, (h,w))

    # color  space conversion  into LAB
    result_lab = np.zeros((h, w, c))
    result_lab[:,:, 0] = 116 * labf(L2/255)-16
    result_lab[:,:, 1] = 500 * (labf(L1/255) - labf(L2/255))
    result_lab[:,:, 2] = 200 * (labf(L2/255) - labf(L3/255))

    return result_lab


# 显示RGB图像
def rgb_show(image, compress_ratio=1):
    height, width, z = image.shape
    x = width/(compress_ratio*100)
    y = height/(compress_ratio*100)

    plt.figure(num='test', figsize=(x, y))
    plt.imshow(image, interpolation='bicubic', vmax=1.0)
    plt.xticks([]), plt.yticks([])  # 隐藏X轴和Y轴的标记位置和labels
    plt.show()


#显示RGB图像
def rgb_image_show(image, width, height,compress_ratio=1):
    x = width/(compress_ratio*100)
    y = height/(compress_ratio*100)
    plt.figure(num='test', figsize=(x, y))
    plt.imshow(image, interpolation='bicubic', vmax=1.0)
    plt.xticks([]), plt.yticks([])  # 隐藏 X轴 和 Y轴的标记位置和labels
    plt.show()

def ycbcr_compress(image, width, height):
    image[:, :, 0]=image[:,:,0]*219/255+16
    image[:, :, 0] = np.clip(image[:, :, 0], 16, 235)
    image[:, :, 1]=image[:,:,1]*224/255+16
    image[:, :, 1] = np.clip(image[:, :, 1], 16, 240)
    image[:, :, 2]=image[:,:,2]*224/255+16
    image[:, :, 2] = np.clip(image[:, :, 2], 16, 240)
    return image
def ycbcr_decompress(image, width, height):
    image[:, :, 0]=(image[:,:,0]-16)*255/219
    image[:, :, 1]=(image[:,:,1]-16)*255/224
    image[:, :, 2]=(image[:,:,2]-16)*255/224
    image = np.clip(image, 0, 255)
    return image
# 0~255 的 Ycbcr转换
def ycbcr2rgb(image, width, height):
    rgb_img = np.zeros(shape=(height, width, 3))
    rgb_img[:,:,0]=image[:,:,0]+1.402*(image[:,:,2]-128) #R= Y+1.402*(Cr-128)
    rgb_img[:,:,1]=image[:,:,0]-0.344136*(image[:,:,1]-128)-0.714136*(image[:,:,2]-128)#G=Y-0.344136*(Cb-128)-0.714136*(Cr-128)
    rgb_img[:,:,2]=image[:,:,0]+1.772*(image[:,:,1]-128) #B=Y+1.772*(Cb-128)
    rgb_img=np.clip(rgb_img, 0, 255)
    return rgb_img
# 0~255 的 Ycbcr转换
def rgb2ycbcr(image, width, height):
    ycbcr_img = np.zeros(shape=(height, width, 3))
    ycbcr_img[:,:,0]=0.299*image[:,:,0]+0.5877*image[:,:,1]+0.114*image[:,:,2]
    ycbcr_img[:,:,1]=128-0.168736*image[:,:,0]-0.331264*image[:,:,1]+0.5*image[:,:,2]
    ycbcr_img[:,:,2]=128+0.5*image[:,:,0]-0.418688*image[:,:,1]-0.081312*image[:,:,2]
    ycbcr_img=np.clip(ycbcr_img, 0, 255)
    return ycbcr_img
def ycbcrshow(image, width, height):
    imagergb=ycbcr2rgb(image, width, height)
    rgb_image_show(imagergb, width, height)

def read_NV12_8_file(filename,width,height):
    #文件长度
    image_bytes=int(width*height*3/2)
    #读出文件
    frame = np.fromfile(filename, count=image_bytes,dtype ="uint8")
    framesize = height * width * 3 // 2  # 一帧图像所含的像素个数
    #算出一半高一半款
    h_h = height // 2
    h_w = width // 2

    Yt = np.zeros(shape=(height, width))
    Cb = np.zeros(shape=(h_h, h_w))
    Cr = np.zeros(shape=(h_h, h_w))
    #读取之后直接做reshape
    Yt[:, :] = np.reshape(frame[0:width*height],newshape=(height,width))
    Cb[:, :] = np.reshape(frame[width * height:image_bytes:2], newshape=(h_h, h_w))
    Cr[:, :] = np.reshape(frame[width * height+1:image_bytes:2], newshape=(h_h, h_w))
    #由420扩展到444
    Cb=Cb.repeat(2, 0)
    Cb=Cb.repeat(2, 1)
    Cr=Cr.repeat(2, 0)
    Cr=Cr.repeat(2, 1)
    #拼接到Ycbcr444
    img = np.zeros(shape=(height, width,3))
    img[:,:,0]= Yt[:, :]
    img[:,:,1]= Cb[:, :]
    img[:,:,2]= Cr[:, :]
    return img


def degamma_srgb(self, clip_range=[0, 1023]):
    # bring data in range 0 to 1
    data = np.clip(self.data, clip_range[0], clip_range[1])
    data = np.divide(data, clip_range[1])

    data = np.asarray(data)
    mask = data > 0.04045

    # basically, if data[x, y, c] > 0.04045, data[x, y, c] = ( (data[x, y, c] + 0.055) / 1.055 ) ^ 2.4
    #            else, data[x, y, c] = data[x, y, c] / 12.92
    data[mask] += 0.055
    data[mask] /= 1.055
    data[mask] **= 2.4

    data[np.invert(mask)] /= 12.92

    # rescale
    return np.clip(data * clip_range[1], clip_range[0], clip_range[1])


def gamma_srgb(data, clip_range=[0, 1023]):
    # bring data in range 0 to 1
    data = np.clip(data, clip_range[0], clip_range[1])
    data = np.divide(data, clip_range[1])

    data = np.asarray(data)
    mask = data > 0.0031308

    # basically, if data[x, y, c] > 0.0031308, data[x, y, c] = 1.055 * ( var_R(i, j) ^ ( 1 / 2.4 ) ) - 0.055
    #            else, data[x, y, c] = data[x, y, c] * 12.92
    data[mask] **= 0.4167
    data[mask] *= 1.055
    data[mask] -= 0.055

    data[np.invert(mask)] *= 12.92

    # rescale
    return np.clip(data * clip_range[1], clip_range[0], clip_range[1])


def get_xyz_reference(cie_version="1931", illuminant="d65"):

    if (cie_version == "1931"):

        xyz_reference_dictionary = {"A" : [109.850, 100.0, 35.585],\
                                    "B" : [99.0927, 100.0, 85.313],\
                                    "C" : [98.074,  100.0, 118.232],\
                                    "d50" : [96.422, 100.0, 82.521],\
                                    "d55" : [95.682, 100.0, 92.149],\
                                    "d65" : [95.047, 100.0, 108.883],\
                                    "d75" : [94.972, 100.0, 122.638],\
                                    "E" : [100.0, 100.0, 100.0],\
                                    "F1" : [92.834, 100.0, 103.665],\
                                    "F2" : [99.187, 100.0, 67.395],\
                                    "F3" : [103.754, 100.0, 49.861],\
                                    "F4" : [109.147, 100.0, 38.813],\
                                    "F5" : [90.872, 100.0, 98.723],\
                                    "F6" : [97.309, 100.0, 60.191],\
                                    "F7" : [95.044, 100.0, 108.755],\
                                    "F8" : [96.413, 100.0, 82.333],\
                                    "F9" : [100.365, 100.0, 67.868],\
                                    "F10" : [96.174, 100.0, 81.712],\
                                    "F11" : [100.966, 100.0, 64.370],\
                                    "F12" : [108.046, 100.0, 39.228]}

    elif (cie_version == "1964"):

        xyz_reference_dictionary = {"A" : [111.144, 100.0, 35.200],\
                                    "B" : [99.178, 100.0, 84.3493],\
                                    "C" : [97.285, 100.0, 116.145],\
                                    "D50" : [96.720, 100.0, 81.427],\
                                    "D55" : [95.799, 100.0, 90.926],\
                                    "D65" : [94.811, 100.0, 107.304],\
                                    "D75" : [94.416, 100.0, 120.641],\
                                    "E" : [100.0, 100.0, 100.0],\
                                    "F1" : [94.791, 100.0, 103.191],\
                                    "F2" : [103.280, 100.0, 69.026],\
                                    "F3" : [108.968, 100.0, 51.965],\
                                    "F4" : [114.961, 100.0, 40.963],\
                                    "F5" : [93.369, 100.0, 98.636],\
                                    "F6" : [102.148, 100.0, 62.074],\
                                    "F7" : [95.792, 100.0, 107.687],\
                                    "F8" : [97.115, 100.0, 81.135],\
                                    "F9" : [102.116, 100.0, 67.826],\
                                    "F10" : [99.001, 100.0, 83.134],\
                                    "F11" : [103.866, 100.0, 65.627],\
                                    "F12" : [111.428, 100.0, 40.353]}

    else:
        print("Warning! cie_version must be 1931 or 1964.")
        return

    return np.divide(xyz_reference_dictionary[illuminant], 100.0)

if __name__ == '__main__':

    img= read_NV12_8_file(filename='NV12_1920(1920)x1080.yuv', height=1080, width=1920)
    print(np.max(img),np.min(img))
    #img =ycbcr_compress(img, height=1080, width=1920)
    #img=ycbcr_decompress(img, height=1080, width=1920)
    rgb=ycbcr2rgb(img, height=1080, width=1920)

    rgb_image_show(rgb/255, height=1080, width=1920, compress_ratio=1)
