import numpy as np
print(np.__version__)

'''
创建
'''
a = np.empty((2,3), np.int) # 参数是Shape和类型
print("a=",a)
b = np.zeros((2,4), np.int)
print("b=",b)
c = np.ones((6,3), np.int)
print("c=",c)
d = np.full((6,3), np.pi)
print("d=",d)
d.shape=(3,6)
print("reshape d=",d)
def func(i,j):
    return i % 4 + 1
e = np.fromfunction(func, (10,4))
print("e=",e)







