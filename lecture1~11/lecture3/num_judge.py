import numpy as np
def num_judge(x, a):  # 对于一个数字如果是3或5的倍数就
    if x % 3 == 0:
        r = 0
    elif x % 5 == 0:
        r = 0
    else:
        r = a
    return r

x = np.linspace(0, 10, 11)
#
y = np.array([num_judge(t, 2) for t in x])#列表生成式

print("y")
numb_judge = np.frompyfunc(num_judge, 2, 1)
y = numb_judge(x,2)
print("y")
