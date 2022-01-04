import numpy as np
a = np.arange(0, 60, 10).reshape(-1, 1) + np.arange(0, 6)
b = np.arange(0, 60, 10)
c = b.reshape(-1,1)
d = np.arange(0, 6)
e = c+d

x, y = np.ogrid[:5, :5]
print(x)
print(y)
z= np.arange(4)
print(z[None, :])
print(z[:, None])