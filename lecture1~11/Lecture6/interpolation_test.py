import cv2
import numpy as np
import matplotlib.pyplot as plt

#临近插值
def Nearest(img, bigger_height, bigger_width, channels):
    near_img = np.zeros(shape=(bigger_height, bigger_width, channels), dtype=np.uint8)

    for i in range(0, bigger_height):
        for j in range(0, bigger_width):
            row = (i / bigger_height) * img.shape[0]
            col = (j / bigger_width) * img.shape[1]
            near_row = int(round(row))
            near_col = int(round(col))
            #取整转到原图的坐标
            #解决边缘问题
            if near_row == img.shape[0] or near_col == img.shape[1]:
                near_row -= 1
                near_col -= 1

            near_img[i][j] = img[near_row][near_col]

    return near_img

#双线性插值
def Bilinear(img, bigger_height, bigger_width, channels):
    bilinear_img = np.zeros(shape=(bigger_height, bigger_width, channels), dtype=np.uint8)

    for i in range(0, bigger_height):
        for j in range(0, bigger_width):
            row = (i / bigger_height) * img.shape[0]
            col = (j / bigger_width) * img.shape[1]
            row_int = int(row)
            col_int = int(col)
            #四个像素里面的坐标
            u = row - row_int
            v = col - col_int
            if row_int == img.shape[0] - 1 or col_int == img.shape[1] - 1:
                row_int -= 1
                col_int -= 1

            bilinear_img[i][j] = (1 - u) * (1 - v) * img[row_int][col_int] + (1 - u) * v * img[row_int][
                col_int + 1] + u * (1 - v) * img[row_int + 1][col_int] + u * v * img[row_int + 1][col_int + 1]

    return bilinear_img


def Bicubic_Bell(num):
    # print( num)
    if -1.5 <= num <= -0.5:
        #  print( -0.5 * ( num + 1.5) ** 2 )
        return -0.5 * (num + 1.5) ** 2
    if -0.5 < num <= 0.5:
        # print( 3/4 - num ** 2 )
        return 3 / 4 - num ** 2
    if 0.5 < num <= 1.5:
        # print( 0.5 * ( num - 1.5 ) ** 2 )
        return 0.5 * (num - 1.5) ** 2
    else:
        # print( 0 )
        return 0

def Bicubic_Spline(num):
    num = np.abs(num)
    if 0 <= num < 1:
        return (2/3) - num ** 2 + (1/2)*num ** 3
    if 1 <= num < 2:
        return (1/6)*(2-num)**3
    else:
        return 0

#双立方插值
def Bicubic(img, bigger_height, bigger_width, channels):
    Bicubic_img = np.zeros(shape=(bigger_height, bigger_width, channels), dtype=np.uint8)

    for i in range(0, bigger_height):
        for j in range(0, bigger_width):
            row = (i / bigger_height) * img.shape[0]
            col = (j / bigger_width) * img.shape[1]
            row_int = int(row)
            col_int = int(col)
            u = row - row_int
            v = col - col_int
            #转换到矩阵里面的坐标
            tmp = 0
            Bicubic_Func=Bicubic_Spline
            for m in range(-1, 3):
                for n in range(-1, 3):
                    if (row_int + m) < 0 or (col_int + n) < 0 or (row_int + m) >= img.shape[0] or (col_int + n) >= \
                            img.shape[1]:
                        row_int = img.shape[0] - 1 - m
                        col_int = img.shape[1] - 1 - n

                    numm = img[row_int + m][col_int + n] * Bicubic_Func(m - u) * Bicubic_Func(n - v)
                    tmp += np.abs(np.trunc(numm))

            Bicubic_img[i][j] = tmp
    return Bicubic_img


if __name__ == '__main__':
    img = cv2.imread('kodim19.png', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img[3][3])
    height, width, channels = img.shape
    print(height, width)

    bigger_height = height + 200
    bigger_width = width + 200
    print(bigger_height, bigger_width)

    near_img = Nearest(img, bigger_height, bigger_width, channels)
    bilinear_img = Bilinear(img, bigger_height, bigger_width, channels)
    Bicubic_img = Bicubic(img, bigger_height, bigger_width, channels)

    plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.title('Source_Image')
    # plt.imshow(img)
    # plt.subplot(2, 2, 2)
    # plt.title('Nearest_Image')
    # plt.imshow(near_img)
    # plt.subplot(2, 2, 3)
    # plt.title('Bilinear_Image')
    # plt.imshow(bilinear_img)
    # plt.subplot(2, 2, 4)
    plt.title('Bicubic_Image')
    plt.imshow(Bicubic_img)
    plt.show()