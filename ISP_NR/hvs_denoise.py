# https://blog.csdn.net/wtzhu_13/article/details/122158378

import numpy as np
import math

import plained_raw
import raw_image_show


def Pixel_color(y, x, bayer_pattern):
    if (bayer_pattern == "RGGB"):
        if (((x % 2) == 0) and ((y % 2) == 0)):
            return "R"
        if (((x % 2) == 1) and ((y % 2) == 0)):
            return "GR"
        if (((x % 2) == 0) and ((y % 2) == 1)):
            return "GB"
        if (((x % 2) == 1) and ((y % 2) == 1)):
            return "B"
    elif (bayer_pattern == "GRBG"):
        if (((x % 2) == 0) and ((y % 2) == 0)):
            return "GR"
        if (((x % 2) == 1) and ((y % 2) == 0)):
            return "R"
        if (((x % 2) == 0) and ((y % 2) == 1)):
            return "B"
        if (((x % 2) == 1) and ((y % 2) == 1)):
            return "GB"
    elif (bayer_pattern == "GBRG"):
        if (((x % 2) == 0) and ((y % 2) == 0)):
            return "GB"
        if (((x % 2) == 1) and ((y % 2) == 0)):
            return "B"
        if (((x % 2) == 0) and ((y % 2) == 1)):
            return "GR"
        if (((x % 2) == 1) and ((y % 2) == 1)):
            return "R"
    elif (bayer_pattern == "BGGR"):
        if (((x % 2) == 0) and ((y % 2) == 0)):
            return "B"
        if (((x % 2) == 1) and ((y % 2) == 0)):
            return "GB"
        if (((x % 2) == 0) and ((y % 2) == 1)):
            return "GR"
        if (((x % 2) == 1) and ((y % 2) == 1)):
            return "B"


def  hvs_behavior_denoise(image, bayer_pattern, initial_noise_level, hvs_min, hvs_max, threshold_red_blue, threshold_green, clip_range, neighborhood_size):
    # Objective: bayer denoising
    # Inputs:
    #   bayer_pattern:  RGGB , GBRG, GRBG, BGGR
    #   initial_noise_level:
    # Output:
    #   denoised bayer raw output
    # Source: Based on paper titled "Noise Reduction for CFA Image Sensors
    #   Exploiting HVS Behaviour," by Angelo Bosco, Sebastiano Battiato,
    #   Arcangelo Bruna and Rosetta Rizzo
    #   Sensors 2009, 9, 1692-1713; doi:10.3390/s90301692

    print("----------------------------------------------------")
    print("Running bayer denoising utilizing hvs behavior...")

    # copy the self.data to raw and we will only work on raw
    # to make sure no change happen to self.data
    raw = np.float32(image)
    raw = np.clip(raw, clip_range[0], clip_range[1])
    height, width  = image.shape

    # First make the bayer_pattern rggb
    # The algorithm is written only for rggb pattern, thus convert all other
    # pattern to rggb. Furthermore, this shuffling does not affect the
    # algorithm output
    # if (bayer_pattern != "rggb"):
    #     raw = utility.helpers(self.data).shuffle_bayer_pattern(bayer_pattern, "rggb")

    # pad two pixels at the border
    no_of_pixel_pad = math.floor(neighborhood_size / 2)  # number of pixels to pad

    raw = np.pad(raw, (no_of_pixel_pad, no_of_pixel_pad), 'reflect')  # reflect would not repeat the border value

    # allocating space for denoised output
    denoised_out = np.empty((height, width), dtype=np.float32)

    texture_degree_debug = np.empty((height, width), dtype=np.float32)

    for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
        for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

            # center pixel
            center_pixel = raw[i, j]

            # signal analyzer block
            half_max = clip_range[1] / 2
            if (center_pixel <= half_max):
                hvs_weight = -(((hvs_max - hvs_min) * center_pixel) / half_max) + hvs_max
            else:
                hvs_weight = (((center_pixel - clip_range[1]) * (hvs_max - hvs_min))/(clip_range[1] - half_max)) + hvs_max

            # noise level estimator previous value
            if (j < no_of_pixel_pad+2):
                noise_level_previous_red   = initial_noise_level
                noise_level_previous_blue  = initial_noise_level
                noise_level_previous_green = initial_noise_level
            else:
                noise_level_previous_green = noise_level_current_green
                if ((i % 2) == 0):  # red
                    noise_level_previous_red = noise_level_current_red
                elif ((i % 2) != 0):  # blue
                    noise_level_previous_blue = noise_level_current_blue

            pxiel_color = Pixel_color(i, j, bayer_pattern)
            # Processings depending on Green or Red/Blue
            # Red
            if (pxiel_color == "R"):
                # get neighborhood
                neighborhood = [raw[i-2, j-2], raw[i-2, j], raw[i-2, j+2],\
                                raw[i, j-2], raw[i, j+2],\
                                raw[i+2, j-2], raw[i+2, j], raw[i+2, j+2]]

                # absolute difference from the center pixel
                d = np.abs(neighborhood - center_pixel)

                # maximum and minimum difference
                d_max = np.max(d)
                d_min = np.min(d)

                # calculate texture_threshold
                texture_threshold = hvs_weight + noise_level_previous_red

                # texture degree analyzer
                if (d_max <= threshold_red_blue):
                    texture_degree = 1.  # 全是类似的
                elif ((d_max > threshold_red_blue) and (d_max <= texture_threshold)):
                    texture_degree = -((d_max - threshold_red_blue) / (texture_threshold - threshold_red_blue)) + 1.  # 不确定根据HVS计算权重
                elif (d_max > texture_threshold):
                    texture_degree = 0.  # 差别大有纹理

                # noise level estimator update
                noise_level_current_red = texture_degree * d_max + (1 - texture_degree) * noise_level_previous_red

            # Blue
            elif (pxiel_color == "B"):
                # get neighborhood
                neighborhood = [raw[i-2, j-2], raw[i-2, j], raw[i-2, j+2],\
                                raw[i, j-2], raw[i, j+2],\
                                raw[i+2, j-2], raw[i+2, j], raw[i+2, j+2]]

                # absolute difference from the center pixel
                d = np.abs(neighborhood - center_pixel)

                # maximum and minimum difference
                d_max = np.max(d)
                d_min = np.min(d)

                # calculate texture_threshold
                texture_threshold = hvs_weight + noise_level_previous_blue

                # texture degree analyzer
                if (d_max <= threshold_red_blue):
                    texture_degree = 1.
                elif ((d_max > threshold_red_blue) and (d_max <= texture_threshold)):
                    texture_degree = -((d_max - threshold_red_blue) / (texture_threshold - threshold_red_blue)) + 1.
                elif (d_max > texture_threshold):
                    texture_degree = 0.

                # noise level estimator update
                noise_level_current_blue = texture_degree * d_max + (1 - texture_degree) * noise_level_previous_blue

            # Green
            elif (pxiel_color=="GR") or (pxiel_color=="GB"):

                neighborhood = [raw[i-2, j-2], raw[i-2, j], raw[i-2, j+2],\
                                raw[i-1, j-1], raw[i-1, j+1],\
                                raw[i, j-2], raw[i, j+2],\
                                raw[i+1, j-1], raw[i+1, j+1],\
                                raw[i+2, j-2], raw[i+2, j], raw[i+2, j+2]]

                # difference from the center pixel
                d = np.abs(neighborhood - center_pixel)

                # maximum and minimum difference
                d_max = np.max(d)
                d_min = np.min(d)

                # calculate texture_threshold
                texture_threshold = hvs_weight + noise_level_previous_green

                # texture degree analyzer
                if (d_max == threshold_green):
                    texture_degree = 1
                elif ((d_max > threshold_green) and (d_max <= texture_threshold)):
                    texture_degree = -(d_max / texture_threshold) + 1.
                elif (d_max > texture_threshold):
                    texture_degree = 0

                # noise level estimator update
                noise_level_current_green = texture_degree * d_max + (1 - texture_degree) * noise_level_previous_green

            # similarity threshold calculation
            if (texture_degree == 1):
                threshold_low = threshold_high = d_max
            elif (texture_degree == 0):
                threshold_low = d_min
                threshold_high = (d_max + d_min) / 2
            elif ((texture_degree > 0) and (texture_degree < 1)):
                threshold_high = (d_max + ((d_max + d_min) / 2)) / 2
                threshold_low = (d_min + threshold_high) / 2

            # weight computation
            weight = np.empty(np.size(d), dtype=np.float32)
            pf = 0.
            for w_i in range(0, np.size(d)):
                if (d[w_i] <= threshold_low):
                    weight[w_i] = 1.
                elif (d[w_i] >= threshold_high):
                    weight[w_i] = 0.
                elif ((d[w_i] > threshold_low) and (d[w_i] < threshold_high)):
                    weight[w_i] = 1. + ((d[w_i] - threshold_low) / (threshold_low - threshold_high))  # 根据值的差异进行权重的计算

                pf += weight[w_i] * neighborhood[w_i] + (1. - weight[w_i]) * center_pixel  # 加权

            denoised_out[i - no_of_pixel_pad, j-no_of_pixel_pad] = pf / np.size(d)  # 平均
            if((pf / np.size(d))>1023):
                print(d)
                print("weight",weight)
                print(i, j, pf/np.size(d), pf, np.size(d), threshold_low, threshold_high)

            # texture_degree_debug is a debug output
            texture_degree_debug[i - no_of_pixel_pad, j-no_of_pixel_pad] = texture_degree

    return np.clip(denoised_out, clip_range[0], clip_range[1]), texture_degree_debug


def interface(raw, width, hight, BayerPatternType, clip_range):
    neighborhood_size = 7
    initial_noise_level = clip_range[1] * 2 / 100

    hvs_min = 20
    hvs_max = 50

    threshold_green = 60
    threshold_red_blue = 100

    raw_, texture_degree_debug = hvs_behavior_denoise(
        raw, BayerPatternType,
        initial_noise_level, hvs_min, hvs_max, threshold_red_blue, threshold_green,
        clip_range, neighborhood_size)

    return raw_


if __name__ == "__main__":
    heigth = 768
    weight = 512

    neighborhood_size = 7
    initial_noise_level = 1023 * 2 / 100

    hvs_min = 20
    hvs_max = 50

    clip_range = [0, 1023]
    threshold_green = 60
    threshold_red_blue = 100

    img = plained_raw.read_plained_file("DSC16_1339_768x512_rggb_blc.raw", height=512, width=768, shift_bits=0)
    img = img.astype(np.float)
    raw_image_show.raw_image_show_fullsize(img/1023, heigth, weight)

    img2, texture_degree_debug = hvs_behavior_denoise(img, "RGGB", initial_noise_level, hvs_min, hvs_max, threshold_red_blue, threshold_green, clip_range, neighborhood_size)
    raw_image_show.raw_image_show_fullsize(img2/1023, heigth, weight)
