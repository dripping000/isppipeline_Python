import numpy as np
print(np.__version__)

'''
创建
'''

print("test1")
a = np.array([1, 2, 3, 4])
b = np.array((5, 6, 7, 8))
c = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(a)

#test2 代码 方便讲解放到一起
a = np.arange(0, 1, 0.1) # 参数是 start end  step
print("a=",a)
b = np.linspace(0, 1, 10) # 参数是 start end  point数量
print("b=",b)
c = np.linspace(0, 1, 10, endpoint=False)# 参数是 start end point数量, 去掉了最后一个点
print("c=",c)
d = np.logspace(0, 2, 5) #
print("log d =",d)


#test3 代码
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







a = np.ones((6,3), np.int)
b = np.linspace(0, 1, 3) # 参数是 start end  point数量
print()
c= a*b
#a.shape=(3,6)
#d= a*b  # 维度错误