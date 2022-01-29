import cv2
import numpy


def gaussian(x, sigma):
    return numpy.exp(-(x**2)/(2*(sigma**2)))/((2*numpy.pi*(sigma**2)))


def distance(x1, y1, x2, y2):
    return numpy.sqrt(numpy.abs((x1-x2)**2+(y1-y2)**2))


def bilateral_filter(image, diameter, sigma_i, sigma_s):
    new_image = numpy.zeros(image.shape)
    image = image.astype(numpy.float)

    # 所有像素
    for row in range(len(image)):
        for col in range(len(image[0])):
            wp_total = 0
            filtered_image = 0

            # 每个窗口的所有像素
            for k in range(diameter):
                for l in range(diameter):
                    # 窗口像素的绝对距离
                    n_x = row - (diameter/2 - k)
                    n_y = col - (diameter/2 - l)

                    if n_x >= len(image):
                        n_x -= len(image)
                    if n_y >= len(image[0]):
                        n_y -= len(image[0])
                    # print(int(n_x), int(n_y), row, col)

                    data = image[int(n_x)][int(n_y)]
                    pixel = image[row][col]

                    # 值域的权重
                    gi = gaussian(data-pixel, sigma_i)
                    # 空间域的权重
                    gs = gaussian(distance(int(n_x), int(n_y), row, col), sigma_s)

                    wp = gi * gs
                    filtered_image = filtered_image + (image[int(n_x)][int(n_y)] * wp)
                    wp_total = wp_total + wp

            filtered_image = filtered_image // wp_total  # 除之后取整
            new_image[row][col] = int(numpy.round(filtered_image))
    return new_image


if __name__ == "__main__":
    image = cv2.imread("lena_gray_noised.bmp", 0)

    # opencv
    filtered_image_OpenCV = cv2.bilateralFilter(image, 7, 10, 10)
    cv2.imwrite("filtered_image_OpenCV.bmp", filtered_image_OpenCV)

    # 自己实现
    image_own = bilateral_filter(image, 7, 10, 10)
    cv2.imwrite("filtered_image_own_bilateral_filter.bmp", image_own)
    print("end")
