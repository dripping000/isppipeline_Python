import numpy as np

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value



a = np.arange(6)
a = a.reshape((2, 3))
b=np.pad(a, 2, pad_with)
c=np.pad(a, 2, pad_with, padder=100)
d=np.pad(a, 2,  'reflect', reflect_type='odd')
e=np.pad(a, 2, 'edge')
print("a=",a)
print("b=",b)
print("c=",c)
print("d=",d)
print("e=",e)
