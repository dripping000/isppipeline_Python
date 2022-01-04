import numpy as np

a = np.array([1, 2, 3, 4], dtype=float)
print(a.dtype)
a = np.array([1, 2, 3, 4])
print(a.dtype)

c = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
print(c.shape)
a = np.array([1, 2, 3, 4])
d = a.reshape((2,2))