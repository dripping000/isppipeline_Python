import numpy as np
#如何定义一个结构体
persontype = np.dtype({ 'names':['name', 'age', 'weight'],'formats':['S30','i', 'f']})
a = np.array([("Zhang", 32, 75.5), ("Wang", 24, 65.2)],dtype=persontype)
print(a[0])