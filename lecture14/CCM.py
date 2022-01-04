import numpy as np
import color_utils as color
import matplotlib.pyplot as plt

import cv2


def rgb2xyz(data, color_space="srgb", clip_range=[0, 65535]):
    # input rgb in range clip_range
    # output xyz is in range 0 to 1

    if (color_space == "srgb"):

        # degamma / linearization
        data = color.degamma_srgb(data,clip_range)
        data = np.float32(data)
        data = np.divide(data, clip_range[1])

        # matrix multiplication`
        output = np.empty(np.shape(data), dtype=np.float32)
        output[:, :, 0] = data[:, :, 0] * 0.4124 + data[:, :, 1] * 0.3576 + data[:, :, 2] * 0.1805
        output[:, :, 1] = data[:, :, 0] * 0.2126 + data[:, :, 1] * 0.7152 + data[:, :, 2] * 0.0722
        output[:, :, 2] = data[:, :, 0] * 0.0193 + data[:, :, 1] * 0.1192 + data[:, :, 2] * 0.9505

    elif (color_space == "adobe-rgb-1998"):

        # degamma / linearization
        data = color.degamma_adobe_rgb_1998(data,clip_range)
        data = np.float32(data)
        data = np.divide(data, clip_range[1])

        # matrix multiplication
        output = np.empty(np.shape(data), dtype=np.float32)
        output[:, :, 0] = data[:, :, 0] * 0.5767309 + data[:, :, 1] * 0.1855540 + data[:, :, 2] * 0.1881852
        output[:, :, 1] = data[:, :, 0] * 0.2973769 + data[:, :, 1] * 0.6273491 + data[:, :, 2] * 0.0752741
        output[:, :, 2] = data[:, :, 0] * 0.0270343 + data[:, :, 1] * 0.0706872 + data[:, :, 2] * 0.9911085

    elif (color_space == "linear"):

        # matrix multiplication`
        output = np.empty(np.shape(data), dtype=np.float32)
        data = np.float32(data)
        data = np.divide(data, clip_range[1])
        output[:, :, 0] = data[:, :, 0] * 0.4124 + data[:, :, 1] * 0.3576 + data[:, :, 2] * 0.1805
        output[:, :, 1] = data[:, :, 0] * 0.2126 + data[:, :, 1] * 0.7152 + data[:, :, 2] * 0.0722
        output[:, :, 2] = data[:, :, 0] * 0.0193 + data[:, :, 1] * 0.1192 + data[:, :, 2] * 0.9505

    else:
        print("Warning! color_space must be srgb or adobe-rgb-1998.")
        return

    return output


def xyz2rgb(data, color_space="srgb", clip_range=[0, 65535]):
    # input xyz is in range 0 to 1
    # output rgb in clip_range

    # allocate space for output
    output = np.empty(np.shape(data), dtype=np.float32)

    if (color_space == "srgb"):

        # matrix multiplication
        output[:, :, 0] = data[:, :, 0] * 3.2406 + data[:, :, 1] * -1.5372 + data[:, :, 2] * -0.4986
        output[:, :, 1] = data[:, :, 0] * -0.9689 + data[:, :, 1] * 1.8758 + data[:, :, 2] * 0.0415
        output[:, :, 2] = data[:, :, 0] * 0.0557 + data[:, :, 1] * -0.2040 + data[:, :, 2] * 1.0570

        # gamma to retain nonlinearity
        output = color.gamma_srgb(data,clip_range)


    elif (color_space == "adobe-rgb-1998"):

        # matrix multiplication
        output[:, :, 0] = data[:, :, 0] * 2.0413690 + data[:, :, 1] * -0.5649464 + data[:, :,2] * -0.3446944
        output[:, :, 1] = data[:, :, 0] * -0.9692660 + data[:, :, 1] * 1.8760108 + data[:, :,2] * 0.0415560
        output[:, :, 2] = data[:, :, 0] * 0.0134474 + data[:, :, 1] * -0.1183897 + data[:, :,2] * 1.0154096


        # gamma to retain nonlinearity
        output =  color.gamma_adobe_rgb_1998(data,clip_range)


    elif (color_space == "linear"):

        # matrix multiplication
        output[:, :, 0] = data[:, :, 0] * 3.2406 + data[:, :, 1] * -1.5372 + data[:, :, 2] * -0.4986
        output[:, :, 1] = data[:, :, 0] * -0.9689 + data[:, :, 1] * 1.8758 + data[:, :, 2] * 0.0415
        output[:, :, 2] = data[:, :, 0] * 0.0557 + data[:, :, 1] * -0.2040 + data[:, :, 2] * 1.0570

        # gamma to retain nonlinearity
        output = output * clip_range[1]

    else:
        print("Warning! color_space must be srgb or adobe-rgb-1998.")
        return

    return output


def xyz2lab(self, cie_version="1931", illuminant="d65"):
    xyz_reference = color.get_xyz_reference(cie_version, illuminant)

    data = self.data
    data[:, :, 0] = data[:, :, 0] / xyz_reference[0]
    data[:, :, 1] = data[:, :, 1] / xyz_reference[1]
    data[:, :, 2] = data[:, :, 2] / xyz_reference[2]

    data = np.asarray(data)

    # if data[x, y, c] > 0.008856, data[x, y, c] = data[x, y, c] ^ (1/3)
    # else, data[x, y, c] = 7.787 * data[x, y, c] + 16/116
    mask = data > 0.008856
    data[mask] **= 1. / 3.
    data[np.invert(mask)] *= 7.787
    data[np.invert(mask)] += 16. / 116.

    data = np.float32(data)
    output = np.empty(np.shape(self.data), dtype=np.float32)
    output[:, :, 0] = 116. * data[:, :, 1] - 16.
    output[:, :, 1] = 500. * (data[:, :, 0] - data[:, :, 1])
    output[:, :, 2] = 200. * (data[:, :, 1] - data[:, :, 2])

    return output


def lab2xyz(self, cie_version="1931", illuminant="d65"):
    output = np.empty(np.shape(self.data), dtype=np.float32)

    output[:, :, 1] = (self.data[:, :, 0] + 16.) / 116.
    output[:, :, 0] = (self.data[:, :, 1] / 500.) + output[:, :, 1]
    output[:, :, 2] = output[:, :, 1] - (self.data[:, :, 2] / 200.)

    # if output[x, y, c] > 0.008856, output[x, y, c] ^ 3
    # else, output[x, y, c] = ( output[x, y, c] - 16/116 ) / 7.787
    output = np.asarray(output)
    mask = output > 0.008856
    output[mask] **= 3.
    output[np.invert(mask)] -= 16 / 116
    output[np.invert(mask)] /= 7.787

    xyz_reference = color.get_xyz_reference(cie_version, illuminant)

    output = np.float32(output)
    output[:, :, 0] = output[:, :, 0] * xyz_reference[0]
    output[:, :, 1] = output[:, :, 1] * xyz_reference[1]
    output[:, :, 2] = output[:, :, 2] * xyz_reference[2]

    return output


def lab2lch(data):
    output = np.empty(np.shape(data), dtype=np.float32)

    output[:, :, 0] = data[:, :, 0]  # L transfers directly
    output[:, :, 1] = np.power(np.power(data[:, :, 1], 2) + np.power(data[:, :, 2], 2), 0.5)
    output[:, :, 2] = np.arctan2(data[:, :, 2], data[:, :, 1]) * 180 / np.pi

    return output


def lch2lab(data):
    output = np.empty(np.shape(data), dtype=np.float32)

    output[:, :, 0] = data[:, :, 0]  # L transfers directly
    output[:, :, 1] = np.multiply(np.cos(data[:, :, 2] * np.pi / 180), data[:, :, 1])
    output[:, :, 2] = np.multiply(np.sin(data[:, :, 2] * np.pi / 180), data[:, :, 1])

    return output


def CCM_convert(data, CCM, color_space="srgb", clip_range=[0, 255]):
    # CCM工作在线性RGB因此需要先进行degamma
    if (color_space == "srgb"):
        data = color.degamma_srgb(data, clip_range)
        data = np.float32(data)
        data = np.divide(data, clip_range[1])  # 归一化
    elif (color_space == "hisi"):
        data = degamma_hisi(data, clip_range, "./gamma_hisi_int.txt")
        data = np.float32(data)
        data = np.divide(data, clip_range[1])  # 归一化

    # matrix multiplication
    output = np.empty(np.shape(data), dtype=np.float32)
    output[:, :, 0] = data[:, :, 0] * CCM[0,0] + data[:, :, 1] * CCM[0,1] + data[:, :, 2] * CCM[0,2]
    output[:, :, 1] = data[:, :, 0] * CCM[1,0] + data[:, :, 1] * CCM[1,1] + data[:, :, 2] * CCM[1,2]
    output[:, :, 2] = data[:, :, 0] * CCM[2,0] + data[:, :, 1] * CCM[2,1] + data[:, :, 2] * CCM[2,2]

    # gamma
    if (color_space == "srgb"):
        output = output*clip_range[1]
        output = color.gamma_srgb(output, clip_range)
    elif (color_space == "hisi"):
        output = output*clip_range[1]
        output = gamma_hisi(output, clip_range, "./gamma_hisi_int.txt")
    return output


def degamma_hisi(data, clip_range, gamma_txt):
    # gamma degamma
    hisi_gamma_x = 1024-1
    hisi_gamma_y = 4096-1

    hisi_degamma_x = 256-1
    hisi_degamma_y = 1.0

    gamma_hisi_x1023_y4095 = []
    degamma_x255_y1 = []

    with open(gamma_txt, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.split(',')

            gamma_hisi_x1023_y4095 = [float(x) for x in line]
            # for j, value in enumerate(line):
            #     print(j, value)

            x = np.arange(0, 1024+1, 1)  # np.arange(start, end+step, step)  [start, end] end/step+1
            plt.plot(x, gamma_hisi_x1023_y4095)
            plt.show()

        for i in range(hisi_degamma_x+1):  # for i in range(0, hisi_degamma_x+1, 1):

            for j, value in enumerate(gamma_hisi_x1023_y4095):
                if (value / hisi_gamma_y * hisi_degamma_x) >= i:
                    degamma_x255_y1.append(j/hisi_gamma_x)
                    break

        x = np.arange(0, hisi_degamma_x+1, 1)
        plt.plot(x, degamma_x255_y1)
        plt.show()

    # degamma
    data = np.clip(data, clip_range[0], clip_range[1])
    data = np.divide(data, clip_range[1])

    height = data.shape[0]
    weight = data.shape[1]
    channels = data.shape[2]

    for row in range(height):               # 遍历高
        for col in range(weight):           # 遍历宽
            pv0 = data[row, col, 0]
            pv1 = data[row, col, 1]
            pv2 = data[row, col, 2]

            data[row, col, 0] = degamma_x255_y1[int(pv0*255)]
            data[row, col, 1] = degamma_x255_y1[int(pv1*255)]
            data[row, col, 2] = degamma_x255_y1[int(pv2*255)]

    data_show = data.copy()
    data_show = np.clip(data_show * clip_range[1], clip_range[0], clip_range[1])
    # gbr = rgb[...,[2,0,1]]
    # data_show = data_show[..., ::-1]
    data_show = data_show[..., [2,1,0]]
    cv2.imshow("data", data_show.astype(np.uint8))
    cv2.waitKey(0)

    return np.clip(data * clip_range[1], clip_range[0], clip_range[1])


def gamma_hisi(data, clip_range, gamma_txt):
    # gamma degamma
    hisi_gamma_x = 1024-1
    hisi_gamma_y = 4096-1

    hisi_degamma_x = 256-1
    hisi_degamma_y = 1.0

    gamma_hisi_x1023_y4095 = []
    degamma_x255_y1 = []

    with open(gamma_txt, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.split(',')

            gamma_hisi_x1023_y4095 = [float(x) for x in line]
            # for j, value in enumerate(line):
            #     print(j, value)

            # x = np.arange(0, 1024+1, 1)  # np.arange(start, end+step, step)  [start, end] end/step+1
            # plt.plot(x, gamma_hisi)
            # plt.show()

        for i in range(hisi_degamma_x+1):  # for i in range(0, hisi_degamma_x+1, 1):

            for j, value in enumerate(gamma_hisi_x1023_y4095):
                if (value / hisi_gamma_y * hisi_degamma_x) >= i:
                    degamma_x255_y1.append(j/hisi_gamma_x)
                    break

        # x = np.arange(0, hisi_degamma_x+1, 1)
        # plt.plot(x, degamma_x255_y1)
        # plt.show()

    # gamma
    data = np.clip(data, clip_range[0], clip_range[1])
    data = np.divide(data, clip_range[1])

    height = data.shape[0]
    weight = data.shape[1]
    channels = data.shape[2]

    for row in range(height):               # 遍历高
        for col in range(weight):           # 遍历宽
            pv0 = data[row, col, 0]
            pv1 = data[row, col, 1]
            pv2 = data[row, col, 2]

            data[row, col, 0] = gamma_hisi_x1023_y4095[int(pv0*1023)] / 4095.0
            data[row, col, 1] = gamma_hisi_x1023_y4095[int(pv1*1023)] / 4095.0
            data[row, col, 2] = gamma_hisi_x1023_y4095[int(pv2*1023)] / 4095.0

    data_show = data.copy()
    data_show = np.clip(data_show * clip_range[1], clip_range[0], clip_range[1])
    # gbr = rgb[...,[2,0,1]]
    # data_show = data_show[..., ::-1]
    data_show = data_show[..., [2,1,0]]
    cv2.imshow("data", data_show.astype(np.uint8))
    cv2.waitKey(0)

    return np.clip(data * clip_range[1], clip_range[0], clip_range[1])


if __name__ == "__main__":
    # CCM = np.array([
    #     [1.507812, -0.546875, 0.039062],
    #     [-0.226562, 1.085938, 0.140625],
    #     [-0.062500, -0.648438, 1.718750],
    #     ])
    CCM = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        ])

    maxvalue = 255
    # image = plt.imread('kodim19.png')
    image = plt.imread('test02.png')
    if (np.max(image) <= 1):
        image = image * maxvalue

    new_image = CCM_convert(image, CCM, color_space="hisi", clip_range=[0, maxvalue])
    color.rgb_show(image / 255)
    color.rgb_show(new_image / 255)
