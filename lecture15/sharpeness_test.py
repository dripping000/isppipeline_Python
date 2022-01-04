from scipy import signal        # for convolutions
from scipy import interpolate        # for convolutions
import numpy as np
import cv2 as cv
import color_utils as color
from matplotlib import pyplot as plt
from scipy import ndimage       # for n-dimensional convolution
def soft_coring(RGB, slope, tau_threshold, gamma_speed):
    # Usage: Used in the unsharp masking sharpening Process
    # Input:
    #   slope:                  controls the boost.
    #                           the amount of sharpening, higher slope
    #                           means more aggresssive sharpening
    #
    #   tau_threshold:          controls the amount of coring.
    #                           threshold value till which the image is
    #                           not sharpened. The lower the value of
    #                           tau_threshold the more frequencies
    #                           goes through the sharpening process
    #
    #   gamma_speed:            controls the speed of convergence to the slope
    #                           smaller value gives a little bit more
    #                           sharpened image, this may be a fine tuner

    return slope * RGB * (1. - np.exp(-((np.abs(RGB / tau_threshold)) ** gamma_speed)))

def gaussian(kernel_size, sigma):

    # calculate which number to where the grid should be
    # remember that, kernel_size[0] is the width of the kernel
    # and kernel_size[1] is the height of the kernel
    temp = np.floor(np.float32(kernel_size) / 2.)

    # create the grid
    # example: if kernel_size = [5, 3], then:
    # x: array([[-2., -1.,  0.,  1.,  2.],
    #           [-2., -1.,  0.,  1.,  2.],
    #           [-2., -1.,  0.,  1.,  2.]])
    # y: array([[-1., -1., -1., -1., -1.],
    #           [ 0.,  0.,  0.,  0.,  0.],
    #           [ 1.,  1.,  1.,  1.,  1.]])
    x, y = np.meshgrid(np.linspace(-temp[0], temp[0], kernel_size[0]),\
                       np.linspace(-temp[1], temp[1], kernel_size[1]))

    # Gaussian equation
    temp = np.exp( -(x**2 + y**2) / (2. * sigma**2) )

    # make kernel sum equal to 1
    return temp / np.sum(temp)

def sharpen_gaussian(RGB, gaussian_kernel_size=[5, 5], gaussian_sigma=2.0,\
                    slope=1.5, tau_threshold=0.02, gamma_speed=4., clip_range=[0, 255]):
    # Objective: sharpen image
    # Input:
    #   gaussian_kernel_size:   dimension of the gaussian blur filter kernel
    #
    #   gaussian_sigma:         spread of the gaussian blur filter kernel
    #                           bigger sigma more sharpening
    #
    #   slope:                  controls the boost.
    #                           the amount of sharpening, higher slope
    #                           means more aggresssive sharpening
    #
    #   tau_threshold:          controls the amount of coring.
    #                           threshold value till which the image is
    #                           not sharpened. The lower the value of
    #                           tau_threshold the more frequencies
    #                           goes through the sharpening process
    #
    #   gamma_speed:            controls the speed of convergence to the slope
    #                           smaller value gives a little bit more
    #                           sharpened image, this may be a fine tuner

    print("----------------------------------------------------")
    print("Running sharpening by unsharp masking...")

    # create gaussian kernel
    gaussian_kernel = gaussian(gaussian_kernel_size, gaussian_sigma)

    # convolove the image with the gaussian kernel
    # first input is the image
    # second input is the kernel
    # output shape will be the same as the first input
    # boundary will be padded by using symmetrical method while convolving
    if np.ndim(RGB > 2):
        image_blur = np.empty(np.shape(RGB), dtype=np.float32)
        for i in range(0, np.shape(RGB)[2]):
            image_blur[:, :, i] = signal.convolve2d(RGB[:, :, i], gaussian_kernel, mode="same", boundary="symm")
    else:
        image_blur = signal.convolove2d(RGB, gaussian_kernel, mode="same", boundary="symm")

    # the high frequency component image
    image_high_pass = RGB - image_blur

    # soft coring (see in utility)
    # basically pass the high pass image via a slightly nonlinear function
    tau_threshold = tau_threshold * clip_range[1]

    # add the soft cored high pass image to the original and clip
    # within range and return
    return np.clip(RGB + soft_coring(image_high_pass, slope, tau_threshold, gamma_speed), clip_range[0], clip_range[1])



def sharpen_convolove(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv.filter2D(image, -1, kernel=kernel)
    return dst

def sharpen_bilateralFilter(RGB):
    d= 5 #kernel size
    sigmaColor= 20 #color domain sigma
    sigmaSpace= 20#space domain sigma
    weight=3
    #weight_ratio=0.3
    h, w, c = RGB.shape
    ycc = color.rgb2ycbcr(RGB, w, h)
    ycc_out=ycc
    y = ycc[:, :, 0]
    cb = ycc[:, :, 1]
    cr = ycc[:, :, 2]
    y_bilateral_filtered = cv.bilateralFilter(y.astype(np.float32) , d, sigmaColor, sigmaSpace)
    detail = ycc[:, :, 0]-y_bilateral_filtered
    y_out =  y_bilateral_filtered + weight * detail
    y_out =  np.clip(y_out,0,255)
    #y_out = (1-weight_ratio)*y_bilateral_filtered + weight_ratio * detail
    ycc_out[:, :, 0] = y_out
    rgb_out = color.ycbcr2rgb(ycc_out, w, h)
    return rgb_out

if __name__ == "__main__":
   #read image
   # read_image
    maxvalue = 255
    LUT_SIZE=17
    pattern = "GRBG"
    image = plt.imread('crop_bgr_0.jpg')
    if (np.max(image) <= 1):
       image = image * maxvalue
    #new_image=sharpen_bilateralFilter(image)
    new_image=sharpen_gaussian(image)
    color.rgb_show(image/255)
    color.rgb_show(new_image/255)
