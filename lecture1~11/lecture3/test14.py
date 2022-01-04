import numpy as np
a = np.arange(10)
print("a=",a)
a[2:4] = 100, 101
b = a[3:7]
b[2] = -10    #b改变a也会改变
print("a=",a)
print("b=",b)
d=a
d[2] = -99
print("a=",a)
print("b=",b)
print("d=",d.astype(int))
