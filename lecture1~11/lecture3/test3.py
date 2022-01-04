import numpy as np

a = np.ones((6,3), np.int)
b = np.linspace(0, 1, 3) # 参数是 start end  point数量
print()
c= a*b
a.shape=(3,6)
#d= a*b  # 维度错误∏