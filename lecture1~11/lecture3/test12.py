import numpy as np
a = np.arange(3)
b = np.arange(10, 13)
#需要注意拼接的维度
v = np.vstack((a, b)) # 按第1轴连接数组
h = np.hstack((a, b)) # 按第0轴连接数组
c = np.column_stack((a, b)) # 按列连接多个一维数组