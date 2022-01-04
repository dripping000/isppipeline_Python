import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2




def trans_gray_PIL(img_dir):
    # using PIL.Image.convert()
    img = Image.open(img_dir)
    img = img.resize((256, 256))
    gray_convert = img.convert('L')
    gray_convert.save('lena_gray_resize.png')
    gray = np.asarray(gray_convert)
    return gray

#kernel_cov可调系数越大越模糊
def NLmeans(img, kernel_cov=5.0):
    width, height = img.shape
    #多大的块去比
    filter_size = 3  # the radio of the filter
    #search windows
    search_size = 10  # the ratio of the search size
    #边缘扩充
    pad_img = np.pad(img, ((filter_size, filter_size), (filter_size, filter_size)), 'symmetric')
    result = np.zeros(img.shape)
    # 归一化的
    kernel = np.ones((2 * filter_size + 1, 2 * filter_size + 1))
    kernel = kernel / ((2 * filter_size + 1) ** 2)

    #遍历所有像素
    for w in range(width):
        for h in range(height):
            #坐标转换
            w1 = w + filter_size
            h1 = h + filter_size
            x_pixels = pad_img[w1-filter_size:w1+filter_size+1, h1-filter_size:h1+filter_size+1]
            # x_pixels = np.reshape(x_pixels, (49, 1)).squeeze()
            w_min = max(w1-search_size, filter_size)
            w_max = min(w1+search_size, width+filter_size-1)
            h_min = max(h1-search_size, filter_size)
            h_max = min(h1+search_size, height+filter_size-1)
            sum_similarity = 0
            sum_pixel = 0
            weight_max = 0
            for x in range(w_min, w_max+1):
                for y in range(h_min, h_max+1):
                    if (x == w1) and (y == h1):
                        continue
                    #y_pixels块
                    y_pixels = pad_img[x-filter_size:x+filter_size+1, y-filter_size:y+filter_size+1]
                    #块中所有点的距离都算出来了
                    distance = x_pixels - y_pixels
                    distance = np.sum(np.multiply(kernel, np.square(distance)))
                    #相似度就是权重
                    similarity = np.exp(-distance/(kernel_cov*kernel_cov))
                    #
                    if similarity > weight_max:
                        weight_max = similarity
                    sum_similarity += similarity
                    sum_pixel += similarity * pad_img[x, y]
            sum_pixel +=  pad_img[w1, h1]
            sum_similarity += 1
            if sum_similarity > 0:
                result[w, h] = sum_pixel / sum_similarity
            else:
                result[w, h] = img[w, h]
    return result




if __name__ == "__main__":
    Img = trans_gray_PIL("nlm_input.png")
    noised_img= Img

    plt.figure(1)
    plt.imshow(noised_img, cmap="gray")
    plt.show()
    #我们自己的实现
    denoised_img = NLmeans(noised_img, 5)
    plt.figure(2)
    plt.imshow(denoised_img, cmap="gray")
    plt.show()
    print("Our NLM")
    plt.figure(3)
    #opencv 快速处理
    denoised_img2 = cv2.fastNlMeansDenoising(noised_img, h=5)
    plt.imshow(denoised_img2, cmap="gray")
    plt.show()
    print("opencv NLM")

