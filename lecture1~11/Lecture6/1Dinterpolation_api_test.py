
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 11)
# x=[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
#原始信号
y = np.sin(x)
#需要插值的点
xnew = np.linspace(0, 10, 101)
plt.plot(x, y, "ro")

for kind in ["nearest", "zero", "slinear", "quadratic", "cubic"]:  # 插值方式
    f = interpolate.interp1d(x, y, kind=kind)
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
    ynew = f(xnew)
    plt.plot(xnew, ynew, label=str(kind))
plt.legend(loc="lower right")
plt.show()

