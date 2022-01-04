from scipy import interpolate
import numpy as np
x1=np.linspace(0,10,20)
y1=np.sin(x1)

sx1=np.linspace(0,12,100)
func1=interpolate.UnivariateSpline(x1,y1,s=0)#强制通过所有点
#fill_value
func2=interpolate.interp1d(x1,y1,kind="cubic",fill_value='extrapolate')#强制通过所有点
func3=interpolate.interp1d(x1,y1,kind="slinear",fill_value='extrapolate')#强制通过所有点
sy1=func1(sx1)
sy2=func2(sx1)
sy3=func3(sx1)
import matplotlib.pyplot as plt
plt.plot(x1,y1,'o')
plt.plot(sx1,sy1)
plt.plot(sx1,sy2)
plt.plot(sx1,sy3)
plt.show()