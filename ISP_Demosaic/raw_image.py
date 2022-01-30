import numpy as np
import raw_image_show as rawshow
import matplotlib.pyplot as plt


#非弱拷贝分离,只是没颜色通道
def simple_separation(image):
    C1=image[::2,::2]
    C2=image[::2,1::2]
    C3=image[1::2,0::2]
    C4=image[1::2,1::2]
    return C1,C2,C3,C4
#带颜色通道的分离
def bayer_channel_separation(data, pattern):
    #------------------------------------------------------
    #   Objective: Outputs four channels of the bayer pattern
    #   Input:
    #       data:   the bayer data
    #       pattern:    RGGB, GBRG, GBRG, or BGGR
    #   Output:
    #       R, GR, GB, B (Quarter resolution images)
    #------------------------------------------------------
    image_data=data
    if (pattern == "RGGB"):
        R = image_data[::2, ::2]
        GR = image_data[::2, 1::2]
        GB = image_data[1::2, ::2]
        B = image_data[1::2, 1::2]
    elif (pattern == "GRBG"):
        GR = image_data[::2, ::2]
        R = image_data[::2, 1::2]
        B = image_data[1::2, ::2]
        GB = image_data[1::2, 1::2]
    elif (pattern == "GBRG"):
        GB = image_data[::2, ::2]
        B = image_data[::2, 1::2]
        R = image_data[1::2, ::2]
        GR = image_data[1::2, 1::2]
    elif (pattern == "BGGR"):
        B = image_data[::2, ::2]
        GB = image_data[::2, 1::2]
        GR = image_data[1::2, ::2]
        R = image_data[1::2, 1::2]
    else:
        print("pattern must be one of :  RGGB, GBRG, GBRG, or BGGR")
        return

    return R, GR, GB, B
#合成
def simple_integration(C1,C2,C3,C4):
    size = np.shape(C1)
    data = np.empty((size[0] * 2, size[1] * 2), dtype=np.float32)
    data[::2, ::2] = C1
    data[::2, 1::2] = C2
    data[1::2, ::2] = C3
    data[1::2, 1::2] = C4
    return data
#按照颜色根据不同的bayer合成
def bayer_channel_integration( R, GR, GB, B, pattern):
        #------------------------------------------------------
        #   Objective: combine data into a raw according to pattern
        #   Input:
        #       R, GR, GB, B:   the four separate channels (Quarter resolution)
        #       pattern:    RGGB, GBRG, GBRG, or BGGR
        #   Output:
        #       data (Full resolution image)
        #------------------------------------------------------
        size = np.shape(R)
        data = np.empty((size[0]*2, size[1]*2), dtype=np.float32)
        # casually use float32,maybe change later
        if (pattern == "RGGB"):
            data[::2, ::2] = R
            data[::2, 1::2] = GR
            data[1::2, ::2] = GB
            data[1::2, 1::2] = B
        elif (pattern == "GRBG"):
            data[::2, ::2] = GR
            data[::2, 1::2] = R
            data[1::2, ::2] = B
            data[1::2, 1::2] = GB
        elif (pattern == "GBRG"):
            data[::2, ::2] = GB
            data[::2, 1::2] = B
            data[1::2, ::2] = R
            data[1::2, 1::2] = GR
        elif (pattern == "bggr"):
            data[::2, ::2] = B
            data[::2, 1::2] = GB
            data[1::2, ::2] = GR
            data[1::2, 1::2] = R
        else:
            print("pattern must be one of these: rggb, grbg, gbrg, bggr")
            return

        return data
#统计直方图
def mono_cumuhistogram(image,max):
    hist,bins=np.histogram(image, bins=range(0,max+1))
    sum=0
    for i in range(0,max):
        sum=sum+hist[i]
        hist[i]=sum
    return hist

def mono_average(image):
    a=np.mean(image)
    return a
#bayer直方图统计
def bayer_cumuhistogram(image,pattern,max):
    R, GR, GB, B=bayer_channel_separation(image,pattern)
    R_hist=mono_cumuhistogram(R,max)
    GR_hist=mono_cumuhistogram(GR,max)
    GB_hist=mono_cumuhistogram(GB,max)
    B_hist=mono_cumuhistogram(B,max)
    return R_hist,GR_hist,GB_hist,B_hist

#bayer的颜色通道的平均值
def bayer_average(image,pattern):
    R, GR, GB, B = bayer_channel_separation(image,pattern)
    R_a = mono_average(R)
    GR_a = mono_average(GR)
    GB_a = mono_average(GB)
    B_a = mono_average(B)
    return R_a, GR_a, GB_a, B_a




#无色顺序统计直方图
def simple_raw_cumuhistogram(image,max):
    C1,C2,C3,C4=simple_separation(image)
    C1_hist=mono_cumuhistogram(C1,max)
    C2_hist=mono_cumuhistogram(C2,max)
    C3_hist=mono_cumuhistogram(C3,max)
    C4_hist=mono_cumuhistogram(C4,max)
    return C1_hist,C2_hist,C3_hist,C4_hist

def simple_raw_average(image):
    C1, C2, C3, C4 = simple_separation(image)
    C1_a = mono_average(C1)
    C2_a = mono_average(C2)
    C3_a = mono_average(C3)
    C4_a = mono_average(C4)
    return C1_a, C2_a, C3_a, C4_a


#获得块
def get_region(image,y,x,h,w):
    region_data= image[y:y+h,x:x+w]
    return region_data

#block和原图需要能够整除,返回值为浮点出
def binning_image(image,height,width,block_size_h,block_size_w):
    region_h_n = int(height / block_size_h)
    region_w_n=int(width / block_size_w)
    binning_image=np.empty((region_h_n*2, region_w_n*2), dtype=np.float32)
    x=0
    y=0
    for j in range(region_h_n):
        for i in range(region_w_n):
            region_data=get_region(image,y,x, block_size_h,block_size_w)
            C1,C2,C3,C4=simple_separation(region_data)
            binning_image[j * 2, i * 2] = np.mean(C1)
            binning_image[j * 2, (i * 2)+1] = np.mean(C2)
            binning_image[(j * 2)+1, i * 2] = np.mean(C3)
            binning_image[(j * 2)+1, (i * 2)+1] = np.mean(C4)
            x=x+block_size_w
        y=y+block_size_h
        x=0
    return binning_image

def test_case_hist():
    b = np.fromfile("RAW_GRBG_plained_4608(9216)x3456_A.raw",dtype ="uint16")
    print("b shape",b.shape)
    print('%#x'%b[0])
    b.shape = [3456, 4608]
    out=b.copy()
    out=out.astype(np.float)
    image=b
    rawshow.raw_image_show_thumbnail(out/1023,3456, 4608)
    hist=mono_cumuhistogram(image, 1023)
    plt.figure(num='hist', figsize=(5, 6))
    plt.bar(range(len(hist)), hist)
    plt.show()
    hist2, bins = np.histogram(image, bins=range(0, 1024))
    plt.figure(num='hist2', figsize=(5, 6))
    plt.bar(range(len(hist)), hist2)
    plt.show()


def test_case_separation_integration():
    b = np.fromfile("RAW_GRBG_plained_4608(9216)x3456_A.raw",dtype ="uint16")
    print("b shape",b.shape)
    print('%#x'%b[0])
    b.shape = [3456, 4608]
    out=b
    rawshow.raw_image_show_thumbnail(out/1023,3456, 4608)

def get_statistcs_test():
    b = np.fromfile("RAW_GRBG_plained_4608(9216)x3456_A.raw",dtype ="uint16")
    print("b shape",b.shape)
    print('%#x'%b[0])
    b.shape = [3456, 4608]
    out=b.copy()
    out=out/1023.0
    rawshow.raw_image_show_thumbnail(out,3456, 4608)
    binning_image_data=binning_image(b, height=3456, width=4608, block_size_h=4, block_size_w=4)
    size=binning_image_data.shape
    rawshow.raw_image_show_thumbnail(binning_image_data/1023, size[1], size[0] )

    print(size)
    #分块
    grid_h_n=54
    grid_w_n=72
    grid_size_h=int(size[0]/54)
    grid_size_w=int(size[1]/72)

    hists = np.empty((grid_h_n, grid_w_n,4,1023), dtype=np.float32)
    averages = np.empty((grid_h_n, grid_w_n, 4), dtype=np.float32)

    for j in range(grid_h_n):
        for i in range(grid_w_n):
            y = int(j * grid_size_h)
            x = int(i * grid_size_w)
            print(x,y,grid_size_h,grid_size_w)
            region_data=get_region(binning_image_data,y,x, grid_size_h,grid_size_w)
            R_hist,GR_hist,GB_hist,B_hist=bayer_cumuhistogram(region_data, "GRBG", 1023)
            hists[j, i, 0, :] = R_hist
            hists[j, i, 1, :] = GR_hist
            hists[j, i, 2, :] = GB_hist
            hists[j, i, 3, :] = B_hist
            R_a, GR_a, GB_a, B_a=bayer_average(region_data,"GRBG")
            averages[j, i, 0] = R_a
            averages[j, i, 1] = GR_a
            averages[j, i, 2] = GB_a
            averages[j, i, 3] = B_a
    rawshow.raw_image_show_fullsize(averages[:,:,0] / 1023, 54, 72)
    return hists,averages

if __name__ == "__main__":
    print ('This is main of module')
    x=test_case_hist()
    x=get_statistcs_test()
    #print(x)
    #get_statistcs_test()