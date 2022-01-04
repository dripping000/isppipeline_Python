
import exposureFusion_ as ef
import matplotlib.pyplot as plt
ef.launch()

"Load Input Images and Setup Dirs"
#------------------------------------------------------------------------------#
path = r"input"

cwd = ef.os.getcwd();
image_stack  = ef.load_images(path)
image_stack  = ef.alignment(image_stack)
#resultsPath = path+"\\results"
resultsPath = cwd+"results"
if ef.os.path.isdir(resultsPath) == True:
    ef.os.chdir(resultsPath)
else:
    ef.os.mkdir(resultsPath); 
    ef.os.chdir(resultsPath)





"Compute Quality Measures"
#------------------------------------------------------------------------------#
#Compute Quality measures multiplied and weighted with weights[x,y,z]
weight_map      = ef.scalar_weight_map(image_stack, weights = [1,1,1])
#weight_map      = ef.scalar_weight_map(image_stack, weights = [0,0,0]) #Performs Pyramid Fusion





"Original Image"
#------------------------------------------------------------------------------#
#load original image i.e center image probably has the median Exposure value(EV)
#filename = ef.os.listdir(path)[len(ef.os.listdir(path))/2]
#original_image = ef.cv2.imread(ef.os.path.join(path, filename), ef.cv2.IMREAD_COLOR)
#ef.cv2.imshow('Original Image', original_image)
###ef.cv2.waitKey(0); ef.cv2.destroyAllWindows()
#ef.cv2.imwrite('img_CenterOriginal.png', original_image.astype('uint8'))





"Naive Exposure Fusion"
#------------------------------------------------------------------------------#
final_imageA, RijA = ef.measures_fusion_naive(image_stack, weight_map)
#ef.cv2.imshow('Naive Fusion', final_imageA.astype('uint8'))
###ef.cv2.waitKey(0); ef.cv2.destroyAllWindows()

plt.figure(2);plt.imshow(final_imageA/255);plt.show()





"Blurred Exposure Fusion"
#------------------------------------------------------------------------------#
final_imageB, RijB = ef.measures_fusion_naive(image_stack, weight_map, blurType = 'gaussian', blurSize = (0,0), blurSigma = 15)
#ef.cv2.imshow('Blur Fusion', final_imageB.astype('uint8'))
###ef.cv2.waitKey(0); ef.cv2.destroyAllWindows()

plt.figure(2);plt.imshow(final_imageB/255);plt.show()




"Bilateral Exposure Fusion"
#------------------------------------------------------------------------------#
final_imageC, RijC = ef.measures_fusion_naive(image_stack, weight_map, blurType = 'bilateral', blurSize = (115,115), blurSigma = 51)
#ef.cv2.imshow('Bilateral Fusion', final_imageC.astype('uint8'))
###ef.cv2.waitKey(0); ef.cv2.destroyAllWindows()

plt.figure(2);plt.imshow(final_imageC/255);plt.show()






"Multiresolution Exposure Fusion"
#------------------------------------------------------------------------------#
final_imageD = ef.measures_fusion_multires(image_stack, weight_map, levels=6)

plt.figure(1);plt.imshow(final_imageD);plt.show()



"simple Exposure Fusion"
#------------------------------------------------------------------------------#
final_imageE = ef.measures_fusion_simple(image_stack)

plt.figure(2);plt.imshow(final_imageE/255);plt.show()



"Display Intermediate Steps and Save"
#------------------------------------------------------------------------------#
#ef.visualize_maps(image_stack, save=False)



"Compute Mean of Image Stack"
#------------------------------------------------------------------------------#
#final_imageE = ef.meanImage(image_stack, save=False)











ef.os.chdir(cwd)








