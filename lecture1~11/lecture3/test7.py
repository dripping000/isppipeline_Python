import numpy as np

b = np.arange(0, 60, 10)
c = b.reshape(-1,1)
print(b[-1])
print(c[-2])  #这个打印和下个打印不一样,是一个行元素
print(c[-2,0])