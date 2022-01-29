# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt

import cv2


def GaussianNoise(src, means, sigma, min_value, max_value):
    NoiseImg = src
    rows = NoiseImg.shape[0]
    cols = NoiseImg.shape[1]

    for i in range(rows):
        for j in range(cols):
            NoiseImg[i, j] = NoiseImg[i, j] + np.random.normal(means, sigma)
            #random.gauss(means, sigma)

    # 避免超过原图值的大小
    return np.clip(NoiseImg, min_value, max_value)


if __name__ == "__main__":
    img = cv2.imread('lena_gray.bmp', 0)

    min_value = 0
    max_value = 255

    mean = 50
    sigma = 0.2
    img1 = GaussianNoise(img, mean, sigma, min_value, max_value)

    # 写到文件
    cv2.imwrite("lena_gray_noised.bmp", img1)
    print("end")
