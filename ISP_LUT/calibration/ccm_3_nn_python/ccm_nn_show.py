import matplotlib.pyplot as plt
import torch
import numpy as np


#%% data
rgb_src = np.loadtxt('src.txt')[:, :3]
rgb_src = torch.from_numpy(rgb_src)
rgb_src = rgb_src.t()
rgb_src = rgb_src.float()

rgb_target = np.loadtxt('target.txt')[:, :3]
rgb_target = torch.from_numpy(rgb_target)
rgb_target = rgb_target.t()
rgb_target = rgb_target.float()


#%% ccm
# res = torch.tensor(
#     [[ 1.8726e+00,-5.4360e-01,-3.2900e-01],
#     [-1.9940e-01, 1.5009e+00,-3.0150e-01],
#     [ 7.0000e-04,-6.1190e-01, 1.6113e+00]]
#         )
# rgb_apply_ccm = res.mm(rgb_src)

ccm_calc1 = -5.4360e-01
ccm_calc2 = -3.2900e-01
ccm_calc3 = -1.9940e-01
ccm_calc5 = -3.0150e-01
ccm_calc6 = 7.0000e-04
ccm_calc7 = -6.1190e-01
rgb_apply_ccm = torch.zeros_like(rgb_src)
rgb_apply_ccm[0, :] = ((1.0 - ccm_calc1 - ccm_calc2) * rgb_src[0, :] + ccm_calc1 * rgb_src[1, :] + ccm_calc2 * rgb_src[2, :])
rgb_apply_ccm[1, :] = (ccm_calc3 * rgb_src[0, :] + (1.0 - ccm_calc3 - ccm_calc5) * rgb_src[1, :] + ccm_calc5 * rgb_src[2, :])
rgb_apply_ccm[2, :] = (ccm_calc6 * rgb_src[0, :] + ccm_calc7 * rgb_src[1, :] + (1.0 - ccm_calc6 - ccm_calc7) * rgb_src[2, :])


#%% show
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.0, 1.0)
ax1.set_zlim(0.0, 1.0)
ax1.set_xlabel('R')
ax1.set_ylabel('G')
ax1.set_zlabel('B')

x2 = rgb_src[0]
y2 = rgb_src[1]
z2 = rgb_src[2]
ax1.scatter(x2, y2, z2, marker='*', c='b', label='origin RGB')

x3 = rgb_target[0]
y3 = rgb_target[1]
z3 = rgb_target[2]
ax1.scatter(x3, y3, z3, marker='o', c='c', label='target rgb')

for i in range(len(x3)):
    ax1.plot([x2[i], x3[i]], [y2[i], y3[i]], [z2[i], z3[i]], 'k-.')
ax1.legend()


fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')

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