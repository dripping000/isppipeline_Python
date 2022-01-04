

"""
Dependencies
"""
###############################################################################
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import color_utils as color
###############################################################################




"""
Program Launch
"""
###############################################################################
def launch():
    print("""
===============================================================================
                                EXPOSURE FUSION
                        Abdulmajeed Muhammad Kabir (2018)
===============================================================================
""")
###############################################################################





"""
Load Images
"""
###############################################################################
def load_images(path, mode='color'):
    """
   FUNCTION: load_images
        Call to load images colored or grayscale and stack them. 
     INPUTS:
        path = location of image
        mode = 'grayscale' or 'colored'
    OUTPUTS:
        read data file
    """
#'-----------------------------------------------------------------------------#
    image_stack = []; i = 0
    for filename in os.listdir(path):
        print("Loading... /" + filename + "...as Image_stack["+str(i)+"]")
        if mode == 'color':
            image = cv2.imread(os.path.join(path, filename))
        else: #mode == 'gray':
            image = cv2.imread(os.path.join(path, filename))
        temp=image[:,:,0].copy()
        image[:, :, 0]=image[:,:,2]
        image[:,:,2]=temp
        image_stack.append(image)
        i += 1
    print("\n")
    return image_stack
###############################################################################




  
"""
Check and Align Images by Size
"""   
###############################################################################
def alignment(image_stack):
    """
   FUNCTION: alignmentent
        Call to Create Uniform Images by adjusting image sizes
     INPUTS:
        image_stack = stack of images from load_images
    OUTPUTS:
        images files of the same size
    """
#'-----------------------------------------------------------------------------#
    sizes = []
    D = len(image_stack)
    for i in range(D):
        sizes.append(np.shape(image_stack[i]))
    sizes = np.array(sizes)
    for i in range(D):
        if np.shape(image_stack[i])[:2] !=  (min(sizes[:,0]),min(sizes[:,1])):
            print("Detected Non-Uniform Sized Image"+str(i)+" ... Resolving ...")
            image_stack[i] = cv2.resize(image_stack[i], (min(sizes[:,1]), min(sizes[:,0])))
            print(" *Done")
    print("\n")
    return image_stack
###############################################################################





"""
Contrast Quality Measure
""" 
###############################################################################
def contrast(image, ksize=1):
    """
   FUNCTION: contrast
        Call to compute the first quality measure: contrast using laplacian kernel
     INPUTS:
        image = input image (colored)
        ksize = 1 means: [[0,1,0],[1,-4,1],[0,1,0]] kernel
    OUTPUTS:
        contrast measure
    """
#'-----------------------------------------------------------------------------#
    image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(image.astype('float64'), cv2.CV_64F, ksize)
    C = cv2.convertScaleAbs(laplacian)
    C = cv2.medianBlur(C.astype('float32') , 5)
    return C.astype('float64')
###############################################################################
    
    



"""
Saturation Quality Measure
"""
###############################################################################
def saturation(image):
    """
   FUNCTION: saturation
        Call to compute second quality measure - st.dev across RGB channels
     INPUTS:
        image = input image (colored)
    OUTPUTS:
        saturation measure
    """
#'-----------------------------------------------------------------------------#
    S = np.std(image, 2)
    return S.astype('float64')
###############################################################################





"""
Well-exposedness Quality Measure
"""  
###############################################################################
def exposedness(image, sigma=0.2):
    """
   FUNCTION: exposedness
        Call to compute third quality measure - exposure using a gaussian curve
     INPUTS:
        image = input image (colored)
        sigma = gaussian curve parameter
    OUTPUTS:
        exposedness measure
    """
#'-----------------------------------------------------------------------------#
    image = cv2.normalize(image, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    gauss_curve = lambda i : np.exp(-((i-0.5)**2) / (2*sigma*sigma))
    R_gauss_curve = gauss_curve(image[:,:,0])
    G_gauss_curve = gauss_curve(image[:,:,1])
    B_gauss_curve = gauss_curve(image[:,:,2])
    E = R_gauss_curve * G_gauss_curve * B_gauss_curve
    return E.astype('float64')
###############################################################################





"""
Scalar Weight Map
""" 
###############################################################################
def scalar_weight_map(image_stack, weights=[1,1,1]):
    """
   FUNCTION: scalar_weight_map
        Call to forcefully "AND"-combine all quality measures defined 
     INPUTS:
        image_measures = stack of quality measures computed for image i 
        image_measures[contrast, saturation, exposedness]
        weights = weight for each quality measure : weights[wc, ws, we]
    OUTPUTS:
        scalar_weight_map for particular image
    """
#'-----------------------------------------------------------------------------#
    H = np.shape(image_stack[0])[0]; 
    W = np.shape(image_stack[0])[1]; 
    D = len(image_stack);
    Wijk = np.zeros((H,W,D), dtype='float64')
    wc = weights[0]
    ws = weights[1]
    we = weights[2]
    print("Computing Weight Maps from Measures using: C=%1.1d, S=%1.1d, E=%1.1d" %(wc,ws,we))
    
    epsilon = 0.000005
    for i in range(D):
        C  = contrast(image_stack[i])
        S  = saturation(image_stack[i])
        E  = exposedness(image_stack[i])
        Wijk[:,:,i] = (np.power(C,wc)*np.power(S,ws)*np.power(E,we)) + epsilon
    normalizer = np.sum(Wijk,2)
    
    for i in range(D):
        Wijk[:,:,i] = np.divide(Wijk[:,:,i], normalizer)
    print(" *Done");print("\n")
    
    return Wijk.astype('float64')
###############################################################################





"""
Naive Measures Fusion
""" 
###############################################################################
def measures_fusion_naive(image_stack, weight_maps, blurType = None, blurSize = (0,0), blurSigma = 15):
    """
   FUNCTION: measures_fusion_naive
        Call to fuse normalized weightmaps and their images
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images 
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        weight_maps = scalar_weight_map for N images
        blurType    = gaussian or bilateral filter applied to weight-map
        blurSize/Sigma = blurring parameters
    OUTPUTS:
        img_fused = single image with fusion of measures
        Rij = fusion of individual images with their weight maps
    """
#'-----------------------------------------------------------------------------#
    H = np.shape(image_stack[0])[0]; 
    W = np.shape(image_stack[0])[1]; 
    D = len(image_stack);
    img_fused = np.zeros((H,W,3), dtype='float64')
    
    if blurType == None:
        print("Performing Naive Blending")
        Rij  = []
        for i in range(D):
            Rijk = image_stack[i]* np.dstack([weight_maps[:,:,i],weight_maps[:,:,i],weight_maps[:,:,i]])
            Rij.append(Rijk)
            img_fused += Rijk
    
    elif blurType == 'gaussian':
        print("Performing Gaussian-Blur Blending")
        Rij  = []
        for i in range(D):
            weight_map = cv2.GaussianBlur(weight_maps[:,:,i], blurSize, blurSigma)
            Rijk = image_stack[i] * np.dstack([weight_map,weight_map,weight_map])
            Rij.append(Rijk)
            img_fused += Rijk
    
    elif blurType == 'bilateral':
        print("Performing Bilateral-Blur Blending")
        Rij  = []
        for i in range(D):
            weight_map = cv2.bilateralFilter(weight_maps[:,:,i].astype('float32'), blurSigma, blurSize[0], blurSize[1])
            Rijk = image_stack[i] * np.dstack([weight_map,weight_map,weight_map])
            Rij.append(Rijk)
            img_fused += Rijk
    print(" *Done");print("\n")
    
    return img_fused, Rij
###############################################################################





"""
Laplacian and Gaussian Pyramids
""" 
###############################################################################
def multires_pyramid(image, weight_map, levels):
    """
   FUNCTION: multires_pyramid
        Call to compute image and weights pyramids
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images 
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        weight_maps = scalar_weight_map for N images
        levels = height of pyramid to use including base pyramid base
    OUTPUTS:
        imgLpyr = list containing image laplacian pyramid
        wGpyr   = list containing weight gaussian pyramid
    """
#'-----------------------------------------------------------------------------#
    levels  = levels - 1
    imgGpyr = [image]
    wGpyr   = [weight_map]
    
    for i in range(levels):
        imgW = np.shape(imgGpyr[i])[1]
        imgH = np.shape(imgGpyr[i])[0]
        imgGpyr.append(cv2.pyrDown(imgGpyr[i].astype('float64')))
        
    for i in range(levels):
        imgW = np.shape(wGpyr[i])[1]
        imgH = np.shape(wGpyr[i])[0]
        wGpyr.append(cv2.pyrDown(wGpyr[i].astype('float64')))

    imgLpyr = [imgGpyr[levels]]
    wLpyr = [wGpyr[levels]]
    
    for i in range(levels, 0, -1):
        imgW = np.shape(imgGpyr[i-1])[1]
        imgH = np.shape(imgGpyr[i-1])[0]
        imgLpyr.append(imgGpyr[i-1] - cv2.resize(cv2.pyrUp(imgGpyr[i]),(imgW,imgH)))
        
    for i in range(levels, 0, -1):
        imgW =  np.shape(wGpyr[i-1])[1]
        imgH = np.shape(wGpyr[i-1])[0]
        wLpyr.append(wGpyr[i-1] - cv2.resize(cv2.pyrUp(wGpyr[i]),(imgW,imgH)))

    return imgLpyr[::-1], wGpyr
###############################################################################





"""
Multiresolution Measures Fusion
""" 
###############################################################################
def measures_fusion_multires(image_stack, weight_maps, levels=6):
    """
   FUNCTION: measures_fusion_multires
        Call to perform multiresolution blending
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images 
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        levels = desired height of the pyramids
        weight_maps = scalar_weight_map for N images
    OUTPUTS:
        finalImage = single exposure fused image
    """
#'-----------------------------------------------------------------------------#
    print("Performing Multiresolution Blending using: "+str(levels)+" Pyramid levels")
    D = np.shape(image_stack)[0]
    
    imgPyramids = []    
    wPyramids = []
    #图像和权重展开
    for i in range(D):
        imgLpyr, wGpyr = multires_pyramid(image_stack[i].astype('float64'), weight_maps[:,:,i], levels)
        imgPyramids.append(imgLpyr)
        wPyramids.append(wGpyr)

    #图像和权重相乘
    blendedPyramids = []
    for i in range(D):
        blended_multires = []
        for j in range(levels):
            blended_multires.append(imgPyramids[i][j] * np.dstack([wPyramids[i][j], wPyramids[i][j], wPyramids[i][j]]))
        blendedPyramids.append(blended_multires)
   #多图融合
    finalPyramid = [] 
    for i in range(levels):
        intermediate = []
        tmp = np.zeros_like(blendedPyramids[0][i])        
        for j in range(D):
            tmp += np.array(blendedPyramids[j][i])
        intermediate.append(tmp)
        finalPyramid.append(intermediate)
     #反向金字塔
    finalImage = []
    blended_final = np.array(finalPyramid[0][0])
    for i in range(levels-1):
        imgH = np.shape(image_stack[0])[0]; 
        imgW = np.shape(image_stack[0])[1]; 
        layerx = cv2.pyrUp(finalPyramid[i+1][0])
        blended_final += cv2.resize(layerx,(imgW,imgH))
    
    blended_final[blended_final < 0] = 0
    blended_final[blended_final > 255] = 255
    finalImage.append(blended_final) 
    print(" *Done"); print("\n")

    return finalImage[0].astype('uint8')
###############################################################################


"""
Multi exposure Measures Fusion
"""


###############################################################################
def measures_fusion_simple(image_stack,max_value=255):

    print("Simple multi exp")
    D,H,W,C = np.shape(image_stack)
    sigma=0.5
    gauss_curve = lambda i: np.exp(-((i - 0.5) ** 2) / (2 * sigma * sigma))

    nromal= round((D-1)/2)
    ycc_out = np.zeros([H,W,C])
    y_out=np.zeros([H, W])
    ycc_weight_sum = np.zeros([H, W])
    for i in range(D):
        RGB=image_stack[i]
        ycc = color.rgb2ycbcr(RGB, W, H)
        y = ycc[:, :, 0]
        weight= gauss_curve(y/max_value)
        y_out = y_out + y*weight
        ycc_weight_sum= ycc_weight_sum+weight
        if i == nromal:
            ycc_out[:, :, 1]=ycc[:, :, 1]
            ycc_out[:, :, 2]=ycc[:, :, 2]

    ycc_out[:, :, 0] = y_out/ycc_weight_sum
    rgb_out = color.ycbcr2rgb(ycc_out, W, H)
    print(" *Done");
    print("\n")

    return rgb_out


###############################################################################


"""
Compute Mean of Image Stack
"""
###############################################################################
def meanImage(image_stack, save=False):
    """
   FUNCTION: meanImage
        Call to perform mean image blending
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images 
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        save = save figures to directory
    OUTPUTS:
        mean of all the images in the stack
    """
#'-----------------------------------------------------------------------------#
    N = len(image_stack)
    H = np.shape(image_stack[0])[0]
    W = np.shape(image_stack[0])[1]
    rr = np.zeros((H,W), dtype='float64')
    gg = np.zeros((H,W), dtype='float64')
    bb = np.zeros((H,W), dtype='float64')
    for i in range(N):
        r, g, b = cv2.split(image_stack[i].astype('float64'))
        rr += r.astype('float64')
        gg += g.astype('float64')
        bb += b.astype('float64')
    MeanImage = np.dstack([rr/N,gg/N,bb/N]).astype('uint8')
    if save == True:
        cv2.imwrite('img_MeanImage.png', MeanImage)
    return MeanImage
###############################################################################





"""
Visualize Image Measures, Weight Maps
"""
###############################################################################
def visualize_maps(image_stack, weights=[1,1,1], save=False):
    """
   FUNCTION: measures_fusion_multires
        Call to perform multiresolution blending
     INPUTS:
        image_stack = lis contains the stack of "exposure-bracketed" images 
        image_stack[img_exposure1, img_exposure2, ... img_exposureN] in order
        weights = importance factor for each measure C,S,E
        save = save figures to directory
    OUTPUTS:
        images of contrast, saturation, exposure, and combined weight for image N
    """
#'-----------------------------------------------------------------------------#
    for N in range(len(image_stack)):
        img_contrast    = contrast(image_stack[N])
        img_saturation  = saturation(image_stack[N])
        img_exposedness = exposedness(image_stack[N])
        #weight_map      = scalar_weight_map([image_stack[N]], weights)
        print("Displaying Measures and Weight Map for Image_stack["+str(N)+"]")
        
        if save == False:
            plt.figure(1);plt.imshow(img_contrast.astype('float'),cmap='gray')
            plt.figure(2);plt.imshow(img_saturation,cmap='gray')
            plt.figure(3);plt.imshow(img_exposedness,cmap='gray')
            #plt.figure(4);plt.imshow(weight_map[:,:,0],cmap='gray') #.astype('uint8')
        else:
            plt.imsave('img_contrast'+str(N)+'.png', img_contrast, cmap = 'gray', dpi=1800)
            plt.imsave('img_saturation'+str(N)+'.png', img_saturation, cmap = 'gray', dpi=1800)
            plt.imsave('img_exposedness'+str(N)+'.png', img_exposedness, cmap = 'gray', dpi=1800)
            #weight_map = 255*cv2.normalize(weight_map[:,:,0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
            #plt.imsave('weightmaps_combined_Normalized'+str(N)+'.png', weight_map.astype('uint8'), cmap = 'gray', dpi=1800)
    print(" *Done"); print("\n")
###############################################################################


   
