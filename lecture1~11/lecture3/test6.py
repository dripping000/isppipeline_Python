import numpy as np
a = np.arange(10)
print("a=",a)
a[2:4] = 100, 101
b = a[3:7]    #浅copy
b[2] = -10    #b改变a也会改变
print("a=",a)
print("b=",b)
c = a[[3, 3, -3, 8]] #深copy
print("c=",c)
c[2] = 100   #C改变a不改变
print("a=",a)
print("c=",c)
d= a[3:7].copy() #深copy
d[2] = -99  #D改变a不改变
print("a=",a)
print("b=",b)
print("d=",d)
d= np.zeros(shape=(4))
d[:]= a[3:7]
d[2] = -99
print("a=",a)
print("b=",b)
print("d=",d.astype(int))
