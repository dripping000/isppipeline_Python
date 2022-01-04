import numpy

a = numpy.array(([3, 2, 1], [2, 5, 7], [4, 7, 8]))
b = numpy.array(([3, 2, 1], [2, 5, 7], [4, 8, 8]))
itemindex = numpy.where((a == 7)|(b>7))
print(itemindex,a[itemindex])
#print(itemindex,a[itemindex[:,0],itemindex[:,1]])