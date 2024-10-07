# -*- coding: utf-8 -*-
import rawpy
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import os


lab_ideal=np.array( # X-Rite官网提供的LAB色彩真值
    [[37.986,13.555,14.059],
      [65.711,18.13,17.81],
      [49.927,-4.88,-21.925],
      [43.139,-13.095,21.905],
      [55.112,8.844,-25.399],
      [70.719,-33.397,-0.199],
      [62.661,36.067,57.096],
      [40.02,10.41,-45.964],
      [51.124,48.239,16.248],
      [30.325,22.976,-21.587],
      [72.532,-23.709,57.255],
      [71.941,19.363,67.857],
      [28.778,14.179,-50.297],
      [55.261,-38.342,31.37],
      [42.101,53.378,28.19],
      [81.733,4.039,79.819],
      [51.935,49.986,-14.574],
      [51.038,-28.631,-28.638],
      [96.539,-0.425,1.186],
      [81.257,-0.638,-0.335],
      [66.766,-0.734,-0.504],
      [50.867,-0.153,-0.27],
      [35.656,-0.421,-1.231],
      [20.461,-0.079,-0.973]],dtype='float32')

M_xyz2rgb=np.array([[3.24096994,-1.53738318,-0.49861076],
                    [-0.96924364,1.8759675,0.04155506],
                    [0.05563008,-0.20397695,1.05697151]])
M_rgb2xyz=np.array([[0.4123908 , 0.35758434, 0.18048079],
                    [0.21263901, 0.71516868, 0.07219231],
                    [0.01933082, 0.11919478, 0.95053216]])

M_rgb2yuv=np.array([[0.299,0.587,0.114],
                    [-0.169,-0.331,0.499],
                    [0.499,-0.418,-0.081]])
M_yuv2rgb=np.array([[9.99999554e-01, -4.46062343e-04,1.40465882],
                     [9.99655449e-01, -3.44551299e-01,-7.15683665e-01],
                     [1.00177531e+00,1.77530689,9.94081794e-04]])


def bayer_demosaic(raw,bayer='RG'):  # 朴素的bayer插值算法
    if bayer=='RG':
        img_r=raw[0::2,0::2]
        img_gr=raw[0::2,1::2]
        img_gb=raw[1::2,0::2]
        img_b=raw[1::2,1::2]
    img=np.dstack((img_r,(img_gr+img_gb)/2,img_b))
    return img
    
def imread(file_path, size=None, bayer='RG', OB=None):  # 图像读取函数
    if os.path.splitext(file_path)[-1] in ('.nef','.NEF'):
        # 读取nef文件，进行OB校正
        H_raw=rawpy.imread(file_path)
        raw=H_raw.raw_image
        OB=H_raw.black_level_per_channel[0]
        white_level=H_raw.camera_white_level_per_channel[0]
        # H_raw.close()
        img=bayer_demosaic(raw,bayer=bayer)
        img[img<OB]=OB
        img=(img-OB).astype('float32')/(white_level-OB)
    elif os.path.splitext(file_path)[-1] in ('.DNG','.dng'):
        # 读取dng文件，进行OB校正
        H_raw=rawpy.imread(file_path)
        raw=H_raw.raw_image
        OB=H_raw.black_level_per_channel[0]
        white_level=H_raw.white_level
        if raw.ndim==2:
            img=bayer_demosaic(raw,bayer=bayer)

        elif raw.ndim==3:
            img=raw[:,:,0:3]
        img[img<OB]=OB
        img=(img-OB).astype('float32')/(white_level-OB)
    elif os.path.splitext(file_path)[-1] in ('.jpg','.JPG','.bmp','.BMP'):
        img=plt.imread(file_path)
        img=img.astype('float32')/255
    elif os.path.splitext(file_path)[-1] in ('.png','.PNG'):
        img=plt.imread(file_path)
        if img.shape[2]==4:
            img=img[:,:,0:3]
    return img


def im2vector(img):  # 将图片转换为向量形式
    size=img.shape
    rgb=np.reshape(img,(size[0]*size[1],3))
    func_reverse=lambda rgb : np.reshape(rgb,(size[0],size[1],size[2]))
    return rgb, func_reverse

def gamma(x,colorspace='sRGB'): #Gamma变换
    y=np. zeros (x. shape)
    y[x>1]=1
    if colorspace in ( 'sRGB', 'srgb'):
        y[(x>=0)&(x<=0.0031308)]=(323/25*x[ (x>=0)&(x<=0.0031308)])
        y[(x<=1)&(x>0.0031308)]=(1.055*abs(x[ (x<=1)&(x>0.0031308)])**(1/2.4)-0.055)
    elif colorspace in ('my'):  
        y[ (x>=0)&(x<=1)]=(1.42*(1-(0.42/(x[(x>=0)&(x<=1)]+0.42))))
    elif colorspace in ('P3'):  
        y[ (x>=0)&(x<=1)]=x[ (x>=0)&(x<=1)]**(1/2.6)
    elif (type(colorspace)==float)|(type(colorspace)==int):
        beta=colorspace
        y[ (x>=0)&(x<=1)]=((1+beta)*(1-(beta/(x[(x>=0)&(x<=1)]+beta))))
    return y

def gamma_reverse(x,colorspace= 'sRGB'): #逆Gamma变换
    y=np.zeros(x.shape)
    y[x>1]=1
    if colorspace in ('sRGB', 'srgb'):
        y[(x>=0)&(x<=0.04045)]=x[(x>=0)&(x<=0.04045)]/12.92
        y[(x>0.04045)&(x<=1)]=((x[(x>0.04045)&(x<=1)]+0.055)/1.055)**2.4
    elif colorspace in ('my'):
        y[(x>=0)&(x<=1)]=0.42/(1-(x[(x>=0)&(x<=1)]/1.42))-0.42         
    return y

def awb(img, awb_para):  #图像做白平衡
    if (img.shape[1]==3)&(img.ndim==2):
        rgb=img
        func_reverse=lambda x : x
    elif (img.shape[2]==3)&(img.ndim==3):
        (rgb,func_reverse)=im2vector(img)
    rgb[:,0]=rgb[:,0]*awb_para[0]
    rgb[:,1]=rgb[:,1]*awb_para[1]
    rgb[:,2]=rgb[:,2]*awb_para[2]
    img=func_reverse(rgb)
    return img

def ccm(img, ccm): #图像进行CCM颜色校正
    if (img.shape[1]==3)&(img.ndim==2):
        rgb=img
        func_reverse=lambda x : x
    elif (img.shape[2]==3)&(img.ndim==3):
        (rgb,func_reverse)=im2vector(img)
    rgb=rgb.transpose()
    rgb=ccm@rgb
    rgb=rgb.transpose()
    img_out=func_reverse(rgb)
    return img_out

def lab2rgb(img,whitepoint='D65'): #lab转rgb
    if (img.ndim==3):
        if (img.shape[2]==3):
            (lab,func_reverse)=im2vector(img)
    elif (img.ndim==2):
        if (img.shape[1]==3):
            lab=img
            func_reverse=lambda x : x
        elif (img.shape[0]>80)&(img.shape[1]>80):
            img=np.dstack((img,img,img))
            (lab,Func_reverse)=im2vector(img)
    lab=lab.transpose()
    if whitepoint=='D65':
        Xn=95.047/100
        Yn=100/100
        Zn=108.883/100
    f_reverse=lambda t : (t>(6/29))*(t**3)+\
        (t<=(6/29))*(3*((6/29)**2)*(t-4/29))
    xyz=np.vstack((Xn*f_reverse((lab[0,:]+16)/116+lab[1,:]/500),
                   Yn*f_reverse((lab[0,:]+16)/116),
                   Zn*f_reverse((lab[0,:]+16)/116-lab[2,:]/200) ))
    rgb=M_xyz2rgb@xyz
    rgb=rgb.transpose()
    rgb=gamma(rgb,colorspace='sRGB')
    rgb_out=func_reverse(rgb)
    return rgb_out

def rgb2lab(img,whitepoint='D65'): #rgb转lab
    if (img.ndim==3):
        if (img.shape[2]==3):
            (rgb,func_reverse)=im2vector(img)
    elif (img.ndim==2):
        if (img.shape[1]==3):
            rgb=img
            func_reverse=lambda x : x
        elif (img.shape[0]>80)&(img.shape[1]>80):
            img=np.dstack((img,img,img))
            (rgb,Func_reverse)=im2vector(img)
    rgb=rgb.transpose()
    rgb=gamma_reverse(rgb,colorspace='sRGB')
    xyz=M_rgb2xyz@rgb
    xyz=xyz.transpose()
    f=lambda t : (t>((6/29)**3))*(t**(1/3))+\
        (t<=(6/29)**3)*(29*29/6/6/3*t+4/29)
    if whitepoint=='D65':
        Xn=95.047/100
        Yn=100/100
        Zn=108.883/100
    L=116*f(xyz[:,1]/Yn)-16
    a=500*(f(xyz[:,0]/Xn)-f(xyz[:,1]/Yn))
    b=200*(f(xyz[:,1]/Yn)-f(xyz[:,2]/Zn))
    Lab=np.vstack((L,a,b)).transpose()
    img_out=func_reverse(Lab)
    return img_out


def impoly(img,poly_position=None): #四边形框选图像ROI
    "(rgb_mean,rgb_std,poly_position)=impoly(img)\n(rgb_mean,rgb_std,poly_position)=impoly(img,poly_position)"
    import matplotlib.pyplot as plt
    if poly_position is None:
        fig=plt.figure(figsize=[12.,7.5],tight_layout=True)
        h_img=plt.imshow(img)
        fig.show()
        # fig.canvas.set_window_title('waiting. ..')
        fig.canvas.manager.set_window_title('waiting. ..')
        pos=plt.ginput(n=4)
        # plt.close(fig)
    else:
        pos=poly_position
    (n,m)=np.meshgrid(np.arange(0.5,6.5)/6,np.arange(0.5,4.5)/4)
    n=n.flatten()
    m=m.flatten()
    x_center=(1-m)*((1-n)*pos[0][0]+n*pos[1][0])+m*(n*pos[2][0]+(1-n)*pos[3][0])
    y_center=(1-m)*((1-n)*pos[0][1]+n*pos[1][1])+m*(n*pos[2][1]+(1-n)*pos[3][1])
    r_sample=min([
        ((pos[0][0]-pos[1][0])**2+(pos[0][1]-pos[1][1])**2)**0.5/6,
        ((pos[1][0]-pos[2][0])**2+(pos[1][1]-pos[2][1])**2)**0.5/4,
        ((pos[2][0]-pos[3][0])**2+(pos[2][1]-pos[3][1])**2)**0.5/6,
        ((pos[3][0]-pos[0][0])**2+(pos[3][1]-pos[0][1])**2)**0.5/4,
        ])*0.2
    if poly_position is None:
        plt.plot(pos[0][0],pos[0][1],'r+')
        plt.plot(pos[1][0],pos[1][1],'r+')
        plt.plot(pos[2][0],pos[2][1],'r+')
        plt.plot(pos[3][0],pos[3][1],'r+')
        # plt.plot(x_center,y_center,'yo')
        plt.plot(x_center-r_sample,y_center-r_sample,'y+')
        plt.plot(x_center+r_sample,y_center-r_sample,'y+')
        plt.plot(x_center-r_sample,y_center+r_sample,'y+')
        plt.plot(x_center+r_sample,y_center+r_sample,'y+')
        plt.draw()
        fig.show()
        poly_position=pos
    else:
        pass
    rgb_mean=np.zeros((24,3))   
    rgb_std=np.zeros((24,3))   
    for block_idx in range(24):
        block=img[int(y_center[block_idx]-r_sample):int(y_center[block_idx]+r_sample),
                  int(x_center[block_idx]-r_sample):int(x_center[block_idx]+r_sample),:]
        rgb_vector,_=im2vector(block)
        rgb_mean[block_idx,:]=rgb_vector.mean(axis=0)
        rgb_std[block_idx,:]=rgb_vector.std(axis=0)
    # plt.close(fig)
    return (rgb_mean,rgb_std,poly_position,h_img,fig)

def func_plot(x): # 绘制动画脚本
    global fig_exist,h_fig,h_ax,f_lab,lab_ideal,h_p,f_obj,h_q
    if not fig_exist:
        h_fig=plt.figure(figsize=(6.2,8),tight_layout=True)
        h_ax=plt.axes(xlim=(-50,60),ylim=(-60,90))
        a,b=np.meshgrid(np.arange(-50,60,0.2),np.arange(90,-60,-0.2))
        L=np.ones(a.shape)*70
        img_back=lab2rgb(np.dstack((L,a,b)))
        plt.imshow(img_back,extent=(-50,60,-60,90))
        plt.plot(lab_ideal[:,1],lab_ideal[:,2],'ks')
        for idx,lab_ideal_el in enumerate(lab_ideal):
            plt.text(lab_ideal_el[1]-5,lab_ideal_el[2]+2,'{}'.format(idx+1))
        h_p=plt.plot(f_lab(x)[:,1],f_lab(x)[:,2],'ko')[0]
        u=f_lab(x)[:,1]-lab_ideal[:,1]
        v=f_lab(x)[:,2]-lab_ideal[:,2]
        h_q=plt.quiver(lab_ideal[:,1],lab_ideal[:,2],u,v,scale_units='xy',scale=1,width=0.003,headwidth=0,headlength=0)
        plt.title('OBJ = {0}'.format(f_obj(x)))
        plt.pause(0.01)
        # h_fig.canvas.draw()
        fig_exist=True
    else:
        plt.sca(h_ax)
        h_p.set_xdata(f_lab(x)[:,1])
        h_p.set_ydata(f_lab(x)[:,2])
        h_q.U=f_lab(x)[:,1]-lab_ideal[:,1]
        h_q.V=f_lab(x)[:,2]-lab_ideal[:,2]
        plt.title('OBJ = {0}'.format(f_obj(x)))
        plt.draw()
        plt.pause(0.01)
        # h_fig.canvas.draw()
        pass


if __name__=='__main__':
    file_path=r'.\\IMG_1548.DNG'


    rgb_ideal=lab2rgb(lab_ideal)
    # np.savetxt('target.txt', rgb_ideal, fmt='%.6f')  # DebugMK

    yuv_ideal=(M_rgb2yuv@gamma_reverse(rgb_ideal).T).T


    img=imread(file_path)
    img_1=img #img_0:OB后的图像
    img_1[img_1<0]=0


    poly_position=None
    (rgb_mean_0,rgb_std,poly_position,h_img,h_fig)=impoly(img,poly_position=poly_position)
    print(rgb_mean_0)


    # AE补偿初始值和AWB自动白平衡
    rgb_mean=rgb_mean_0

    # 最优化方法计算最佳曝光补偿
    func_ae=lambda ae_comp : np.prod(gamma(ae_comp*rgb_mean[19:23,1],colorspace='sRGB'))/np.prod(rgb_ideal[19:23,1])-1
    ae_res=optimize.root_scalar(func_ae, bracket=[0, 100], method='brentq')
    ae_comp_init=ae_res.root
    # 单独计算22色块的曝光补偿
    # ae_comp=gamma_reverse(rgb_ideal,colorspace='sRGB')[20,1]/rgb_mean[20,1]
    ae_comp=gamma_reverse(rgb_ideal,colorspace='sRGB')[21,1]/rgb_mean[21,1]
    # ae_comp=gamma_reverse(rgb_ideal,colorspace='sRGB')[22,1]/rgb_mean[22,1]
    ae_comp_init=ae_comp
    print('AE补偿初始值:',ae_comp_init)

    awb_para=[rgb_mean[21,1]/rgb_mean[21,0],1,rgb_mean[21,1]/rgb_mean[21,2]]
    rgb_mean=awb(rgb_mean,awb_para)


    x2ccm=lambda x : np.array([[1-x[0]-x[1],x[0],x[1]],
                                [x[2],1-x[2]-x[3],x[3]],
                                [x[4],x[5],1-x[4]-x[5]]])

    f_lab=lambda x : rgb2lab(gamma(ccm(rgb_mean*x[6],x2ccm(x)),colorspace='sRGB'))
    f_error=lambda x : f_lab(x)-lab_ideal
    f_DeltaE=lambda x : np.sqrt((f_error(x)**2).sum(axis=1,keepdims=True)).mean()
    f_DeltaC=lambda x : np.sqrt((f_error(x)[:,1:]**2).sum(axis=1,keepdims=True)).mean()
    f_obj=lambda x : f_DeltaE(x)
    x0=np.array([0,0,0,0,0,0,ae_comp_init])
    print('初始值:',round(f_DeltaE(x0),4))
    func=lambda x : print('',f_obj(x))
    fig_exist=False
    result=optimize.minimize(f_obj,x0,method='Powell',callback=func_plot)
    print('==最优解==')
    print('最优值:',round(f_obj(result.x),4))
    ae_comp=result.x[6]
    print('AE补偿:',ae_comp)
    print('AWB Gain = :',awb_para)
    print('CCM:')
    print(np.round(x2ccm(result.x),4))


    rgb_src = rgb_mean*ae_comp
    np.savetxt('src.txt', rgb_src, fmt='%.6f')
    rgb_src = np.loadtxt('src.txt')[:, :3]

    rgb_target = ccm(rgb_src,x2ccm(result.x))
    np.savetxt('target.txt', rgb_target, fmt='%.6f')
    rgb_target = np.loadtxt('target.txt')[:, :3]


    img_opti=gamma(ccm(awb(img*ae_comp,awb_para),x2ccm(result.x)),colorspace='sRGB')
    h_img.set_array(img_opti)
    h_fig.canvas.flush_events()

    # ccm_DebugMK = np.array([[ 0.8966, -0.1853,  0.2887],
    #                         [-0.2177,  0.9146,  0.3031],
    #                         [-0.1101, -0.2234,  1.3335]])
    # img_opti_=gamma(ccm(awb(img*ae_comp,awb_para),ccm_DebugMK),colorspace='sRGB')

    plt.subplots(2,2,squeeze=False,tight_layout=True)
    plt.subplot(2,2,1,xticks=[],yticks=[])
    plt.imshow(img)
    plt.title('RAW')
    plt.subplot(2,2,2,xticks=[],yticks=[])
    plt.imshow(awb(img*ae_comp,awb_para))
    plt.title('AWB')
    plt.subplot(2,2,3,xticks=[],yticks=[])
    plt.imshow(gamma(awb(img*ae_comp,awb_para)))
    plt.title('Gamma')
    plt.subplot(2,2,4,xticks=[],yticks=[])
    plt.imshow(img_opti)
    plt.title('CCM-Gamma')
    plt.savefig('ccm.png', dpi=300, bbox_inches='tight')

    print(np.hstack((np.arange(1,25).reshape((24,1)),np.around(f_error(result.x),2))))


    pass