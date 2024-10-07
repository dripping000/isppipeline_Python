'''
https://zhuanlan.zhihu.com/p/108626480/
https://blog.csdn.net/qq_37164776/article/details/126832303

https://zhuanlan.zhihu.com/p/413851281
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
import numpy as np


#%% data
rgb_src = np.loadtxt('src.txt')[:, :3]
rgb_target = np.loadtxt('target.txt')[:, :3]

rgb_data=torch.from_numpy(rgb_src)*255.0
rgb_data=rgb_data.t()
rgb_data = rgb_data.float()

rgb_target=torch.from_numpy(rgb_target)*255.0
rgb_target=rgb_target.t()
rgb_target = rgb_target.float()


#%% ccm nn
ccm_calc1 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc2 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc3 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc5 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc6 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
ccm_calc7 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

def squared_loss(rgb_tmp, rgb_ideal):
    print("loss: " + str(torch.sum((rgb_tmp-rgb_ideal)**2)))
    return torch.sum((rgb_tmp-rgb_ideal)**2)

def sgd(params, lr, batch_size):
    for param in params:
        # DebugMK 通过batch_size控制step步长在0.01左右
        print("step: " + str(lr * param.grad/batch_size) + " param.grad: " + str(param.grad) + " lr: " + str(lr) + " batch_size: " + str(batch_size))
        print("param.data(before): " + str(param.data))
        param.data -= lr * param.grad/batch_size;
        print("param.data(after): " + str(param.data))

def net(ccm_calc1, ccm_calc2, ccm_calc3, ccm_calc5, ccm_calc6, ccm_calc7, rgb_data):
    rgb_tmp = torch.zeros_like(rgb_data)
    # 量化1024
    # rgb_tmp[0, :] = ((1024.0 - ccm_calc1 - ccm_calc2) * rgb_data[0, :] + ccm_calc1 * rgb_data[1, :] + ccm_calc2 * rgb_data[2, :]) / 1024.0
    # rgb_tmp[1, :] = (ccm_calc3 * rgb_data[0, :] + (1024.0 - ccm_calc3 - ccm_calc5) * rgb_data[1, :] + ccm_calc5 * rgb_data[2, :]) / 1024.0
    # rgb_tmp[2, :] = (ccm_calc6 * rgb_data[0, :] + ccm_calc7 * rgb_data[1, :] + (1024.0 - ccm_calc6 - ccm_calc7) * rgb_data[2, :]) / 1024.0

    rgb_tmp[0, :] = ((1.0 - ccm_calc1 - ccm_calc2) * rgb_data[0, :] + ccm_calc1 * rgb_data[1, :] + ccm_calc2 * rgb_data[2, :])
    rgb_tmp[1, :] = (ccm_calc3 * rgb_data[0, :] + (1.0 - ccm_calc3 - ccm_calc5) * rgb_data[1, :] + ccm_calc5 * rgb_data[2, :])
    rgb_tmp[2, :] = (ccm_calc6 * rgb_data[0, :] + ccm_calc7 * rgb_data[1, :] + (1.0 - ccm_calc6 - ccm_calc7) * rgb_data[2, :])
    return rgb_tmp

lr = 3
num_epochs = 1000
for epoch in range(num_epochs):
    l = squared_loss(net(ccm_calc1, ccm_calc2, ccm_calc3, ccm_calc5, ccm_calc6, ccm_calc7, rgb_data), rgb_target)
    l.backward()
    sgd([ccm_calc1, ccm_calc2, ccm_calc3, ccm_calc5, ccm_calc6, ccm_calc7], lr, 100)
    ccm_calc1.grad.data.zero_()
    ccm_calc2.grad.data.zero_()
    ccm_calc3.grad.data.zero_()
    ccm_calc5.grad.data.zero_()
    ccm_calc6.grad.data.zero_()
    ccm_calc7.grad.data.zero_()
    print('epoch %d, loss %f'%(epoch, l))


#%% ccm result
# 量化1024
# res = torch.tensor([[1024.0 - ccm_calc1 - ccm_calc2, ccm_calc1, ccm_calc2],
#                     [ccm_calc3, 1024.0-ccm_calc3-ccm_calc5, ccm_calc5],
#                     [ccm_calc6, ccm_calc7, 1024.0-ccm_calc6-ccm_calc7]], dtype=torch.float32)
# print(res/1024);
# rgb_apply_ccm = res.mm(rgb_data)/1024.0

res = torch.tensor([[1.0 - ccm_calc1 - ccm_calc2, ccm_calc1, ccm_calc2],
                    [ccm_calc3, 1.0-ccm_calc3-ccm_calc5, ccm_calc5],
                    [ccm_calc6, ccm_calc7, 1.0-ccm_calc6-ccm_calc7]], dtype=torch.float32)
print(res);
rgb_apply_ccm = res.mm(rgb_data)


#%% show
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')

x2 = rgb_data[0]
y2 = rgb_data[1]
z2 = rgb_data[2]

ax1.scatter(x2, y2, z2, marker='*', c='b', label='origin RGB')

ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.0, 1.0)
ax1.set_zlim(0.0, 1.0)
ax1.set_xlabel('R')
ax1.set_ylabel('G')
ax1.set_zlabel('B')

x3 = rgb_target[0]
y3 = rgb_target[1]
z3 = rgb_target[2]
ax1.scatter(x3, y3, z3, marker='o', c='c', label='target rgb')

for i in range(len(x3)):
    ax1.plot([x2[i], x3[i]], [y2[i], y3[i]], [z2[i], z3[i]], 'k-.')
ax1.legend()

ax2.set_xlim(0.0, 1.0)
ax2.set_ylim(0.0, 1.0)
ax2.set_zlim(0.0, 1.0)
ax2.set_xlabel('R')
ax2.set_ylabel('G')
ax2.set_zlabel('B')
ax2.scatter(x3, y3, z3, marker='o', c='c', label='target rgb')

x4 = rgb_apply_ccm[0]
y4 = rgb_apply_ccm[1]
z4 = rgb_apply_ccm[2]
ax2.scatter(x4, y4, z4, marker='^', c='b', label='apply ccm rgb')

for i in range(len(x3)):
    ax2.plot([x3[i], x4[i]], [y3[i], y4[i]], [z3[i], z4[i]], 'k-.')
ax2.legend()

plt.show()

pass