import math
import f_test as ftest

'''
变量类型
'''
counter = 100  # 赋值整型变量
miles = 1000.0  # 浮点型
name = "John"  # 字符串
print(counter,miles,name)
print(type(counter),type(miles),type(name))



'''
Range
'''
range1=range(10)        # 从 0 开始到 10
print(range1)
range2=range(1, 11)     # 从 1 开始到 11
print(range2)
range3=range(0, 30, 5)  # 步长为 5
print(range3)
range4=range(0, 10, 3)  # 步长为 3
print(range4)
range5=range(0, -10, -1) # 负数
print(range5)
range6=range(0)
print(range6)
range7=range(1, 0)
print(range7)

for i in range(10):
    print(i)
'''
module 测试
'''
x, y = ftest.move(100, 100, 60, math.pi / 6)
r = ftest.move(100, 100, 60, math.pi / 6)
print(x, y)
print(r)
