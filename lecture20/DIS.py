import cv2
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import cvxpy as cp
#SIFT 算法是个很关键的算法
sift = cv2.xfeatures2d.SIFT_create(200)
SetDispThresh=int(30)

def getAffMat(I1, I2):
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Finding sift features
    kp1, desc1 = sift.detectAndCompute(I1, None)
    kp2, desc2 = sift.detectAndCompute(I2, None)

    # Finding good matches using ratio testing
    #Brute-Force匹配器
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    # 将不满足的最近邻的匹配之间距离比率大于设定的阈值匹配剔除。
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    pts_src = []
    pts_dst = []
    for i in range(len(good)):
        pts_src.append([kp1[good[i].queryIdx].pt[0], kp1[good[i].queryIdx].pt[1]])
        pts_dst.append([kp2[good[i].trainIdx].pt[0], kp2[good[i].trainIdx].pt[1]])

    pts_src = np.array(pts_src).astype(np.float32)
    pts_dst = np.array(pts_dst).astype(np.float32)

    # Computing affine matrix using the best matches
    #计算转换矩阵
    return cv2.estimateRigidTransform(pts_src, pts_dst, fullAffine=False)


v = cv2.VideoCapture("vid1.mp4")
n_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))

# Generating the Xdata and Ydata
transforms = [[], [], [], []]
count = 0
while v.isOpened():

    ret, frame = v.read()

    if ret == True:

        if count > 0:
            transMat = getAffMat(prev, frame)
            try:
                transforms[0].append(transMat[0][2])
                transforms[1].append(transMat[1][2])
                transforms[2].append(np.arctan2(transMat[1][0], transMat[0][0]))
                transforms[3].append(np.sqrt(transMat[1][0] ** 2 + transMat[0][0] ** 2))
            except:
                transforms[0].append(0)
                transforms[1].append(0)
                transforms[2].append(0)
                transforms[3].append(1)

        count += 1
        prev = frame
        print(str((count / n_frames) * 100) + "% completed")
    else:
        break

v.release()

# Storing the data
with open('transforms.pkl', 'wb') as f:
    pickle.dump(transforms, f)

# Loading the data
with open('transforms.pkl', 'rb') as f:
	transforms = pickle.load(f)

# Computing the trajectory
trajectory = np.cumsum(transforms, axis=1)
with open('trajectory.pkl', 'wb') as f:
	pickle.dump(trajectory, f)

# Smoothening the trajectories
# fx is the optimal x trajectory
# fy is the optimal y trajectory
# fth is the optimal theta trajectory
# fs is the optimal scale trajectory
fx = cp.Variable(len(trajectory[0]))
fy = cp.Variable(len(trajectory[1]))
fth = cp.Variable(len(trajectory[2]))
fs = cp.Variable(len(trajectory[3]))

lbd1 = 10000
lbd2 = 1000
lbd3 = 100000
DispThresh = int(30)
constraints = [cp.abs(fx - trajectory[0]) <= DispThresh,
 			   cp.abs(fy - trajectory[1]) <= DispThresh,
 			   cp.abs(fth - trajectory[2]) <= 0.05,
 			   cp.abs(fs - trajectory[3]) <= 0.01]

# Defining the minimization objective function
obj = 0
#变化控制,原论文中没有
#for i in range(len(trajectory[0])):
#	obj += ( (trajectory[0][i]-fx[i])**2 + (trajectory[1][i]-fy[i])**2 + (trajectory[2][i]-fth[i])**2 + (trajectory[3][i]-fs[i])**2 )

#距离差 帧2-帧1
# DP1
for i in range(len(trajectory[0])-1):
	obj += lbd1*(cp.abs(fx[i+1]-fx[i]) + cp.abs(fy[i+1]-fy[i]) + cp.abs(fth[i+1]-fth[i]) + cp.abs(fs[i+1]-fs[i]))
#速度差  (帧3-帧2)-(帧2-帧1)
# DP2
for i in range(len(trajectory[0])-2):
	obj += lbd2*(cp.abs(fx[i+2]-2*fx[i+1]+fx[i]) + cp.abs(fy[i+2]-2*fy[i+1]+fy[i]) + cp.abs(fth[i+2]-2*fth[i+1]+fth[i]) + cp.abs(fs[i+2]-2*fs[i+1]+fs[i]))
#加速度差  ((帧4-帧3)-(帧3-帧2))-((帧3-帧2)-(帧2-帧1))
# DP3
for i in range(len(trajectory[0])-3):
	obj += lbd3*(cp.abs(fx[i+3]-3*fx[i+2] + 3*fx[i+1]-fx[i]) + cp.abs(fy[i+3]-3*fy[i+2]+3*fy[i+1]-fy[i]) + cp.abs(fth[i+3]-3*fth[i+2]+3*fth[i+1]-fth[i]) + cp.abs(fs[i+3]-3*fs[i+2]+3*fs[i+1]-fs[i]))

prob = cp.Problem(cp.Minimize(obj), constraints)
print("Started Solving the optimization using ECOS")
prob.solve(solver=cp.ECOS)
print("Optimization solved")

# Results
plt.subplot(2, 2, 1)
plt.plot(trajectory[0])
plt.plot(fx.value)
plt.title("Horizontal trajectory")
plt.subplot(2, 2, 2)
plt.plot(trajectory[1])
plt.plot(fy.value)
plt.title("Verical trajectory")
plt.subplot(2, 2, 3)
plt.plot(trajectory[2])
plt.plot(fth.value)
plt.title("Angle of rotn. trajectory")
plt.subplot(2, 2, 4)
plt.plot(trajectory[3])
plt.plot(fs.value)
plt.title("Scale trajectory")
plt.show()

smoothTrajectory = np.array([fx.value, fy.value, fth.value, fs.value])

# Storing smooth trajectory
with open('smoothTrajectory.pkl', 'wb') as f:
	pickle.dump(smoothTrajectory, f)


def getAngle(a, b, c):
    a = np.array([a[0], a[1]])
    b = np.array([b[0], b[1]])
    c = np.array([c[0], c[1]])

    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle


# Loading the data
with open('transforms.pkl', 'rb') as f:
    transforms = pickle.load(f)

with open('trajectory.pkl', 'rb') as f:
    trajectory = pickle.load(f)

with open('smoothTrajectory.pkl', 'rb') as f:
    smoothTrajectory = pickle.load(f)

# Box trajectory
difference = trajectory - smoothTrajectory

# Frame transform
smoothTransforms = transforms - difference

# Creating output video
v = cv2.VideoCapture("vid1.mp4")
W = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(v.get(cv2.CAP_PROP_FPS))
n_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
DispThresh = int(30)
DispThresh += 10

OrigBox = [[DispThresh, W - DispThresh, W - DispThresh, DispThresh],
           [DispThresh, DispThresh, H - DispThresh, H - DispThresh],
           [1, 1, 1, 1]]

out = cv2.VideoWriter('video_out.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (W - 2 * DispThresh, H - 2 * DispThresh))
m = np.zeros([2, 3])

for i in range(n_frames - 1):
    ret, frame = v.read()
    if ret == True:

        # Writing to output file
        m[0][0] = (smoothTransforms[3][i]) * np.cos(smoothTransforms[2][i])
        m[0][1] = -(smoothTransforms[3][i]) * np.sin(smoothTransforms[2][i])
        m[1][0] = (smoothTransforms[3][i]) * np.sin(smoothTransforms[2][i])
        m[1][1] = (smoothTransforms[3][i]) * np.cos(smoothTransforms[2][i])
        m[0][2] = smoothTransforms[0][i]
        m[1][2] = smoothTransforms[1][i]
        #矫正图像
        stable = cv2.warpAffine(frame, m, (W, H))
        #裁剪
        boxPart = stable[DispThresh:H - DispThresh, DispThresh:W - DispThresh, :]
        out.write(boxPart)

        # Displaying the difference
        m[0][0] = (1 + difference[3][i - 1]) * np.cos(difference[2][i - 1])
        m[0][1] = -(1 + difference[3][i - 1]) * np.sin(difference[2][i - 1])
        m[1][0] = (1 + difference[3][i - 1]) * np.sin(difference[2][i - 1])
        m[1][1] = (1 + difference[3][i - 1]) * np.cos(difference[2][i - 1])
        m[0][2] = difference[0][i - 1]
        m[1][2] = difference[1][i - 1]
        recPts = np.matmul(m, OrigBox)

        pt0 = (int(round(recPts[0][0])), int(round(recPts[1][0])))
        pt1 = (int(round(recPts[0][1])), int(round(recPts[1][1])))
        pt2 = (int(round(recPts[0][2])), int(round(recPts[1][2])))
        pt3 = (int(round(recPts[0][3])), int(round(recPts[1][3])))
        #在原有帧上画出裁剪框
        white = (255, 255, 255)
        cv2.line(frame, pt0, pt1, white, 2)
        cv2.line(frame, pt1, pt2, white, 2)
        cv2.line(frame, pt2, pt3, white, 2)
        cv2.line(frame, pt3, pt0, white, 2)
        #将两帧缩小到一样大,然后拼接
        frame = cv2.resize(frame, (W - 2 * DispThresh, H - 2 * DispThresh))
        dispFrame = cv2.hconcat([frame, boxPart])
        #动态显示帧
        # If output too wide
        if dispFrame.shape[1] >= 1920:
            dispFrame = cv2.resize(dispFrame, (dispFrame.shape[1] // 2, dispFrame.shape[0] // 2));
        cv2.imshow('DispFrame', dispFrame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
v.release()
out.release()