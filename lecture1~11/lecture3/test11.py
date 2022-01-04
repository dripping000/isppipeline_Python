import numpy as np
#随机数的例子
np.random.seed(42)
a = np.random.randint(0,10,size=(4,5))
print("a=",a)
print("np.sum(a, axis=1)=",np.sum(a, axis=1))
print("np.sum(a, axis=0)=",np.sum(a, axis=0))
print("np.sum(a,1,keepdims=True)=",np.sum(a,1,keepdims=True))
print("np.sum(a,0,keepdims=True)=",np.sum(a,0,keepdims=True))

#又有广播
a = np.array([1, 3, 5, 7])
b = np.array([2, 4, 6])
#b[:, None],a[None, :] 相当于reshape
print(np.maximum(a[None, :], b[:, None]))#maxinum返回两组矩阵广播计算后的结果