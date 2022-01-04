import numpy as np

a = np.array([6, 3, 7, 4, 6, 9, 2, 6, 7, 4, 3, 7])
b = np.array([ 1,  3,  6,  9, 10])
print(np.split(a, b))# 按元素位置进行分段
print(np.split(a, 2))