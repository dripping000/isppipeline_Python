import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np
import plained_raw
import raw_image_show
import raw_image

def simple_blc(img,blacklevel):
    img=img.astype(np.int16)
    img= img - blacklevel
    img=np.clip(img,a_min=0,a_max=np.max(img))
    return img

def blc_process(img,blacklevel1,blacklevel2,blacklevel3,blacklevel4):
    img = img.astype(np.int16)
    C1, C2, C3, C4 = raw_image.simple_separation(img)
    C1 = C1 - blacklevel1
    C2 = C2 - blacklevel2
    C3 = C3 - blacklevel3
    C4 = C4 - blacklevel4
    img = raw_image.simple_integration(C1, C2, C3, C4)
    img=np.clip(img,a_min=0,a_max=np.max(img))
    return img

def block_blc(img,blacklevel1,blacklevel2,blacklevel3,blacklevel4):
    img = img.astype(np.int16)
    C1,C2,C3,C4=raw_image.simple_separation(img)
    size=C1.shape
    size_new=(size[1],size[0])
    blacklevel1=cv2.resize(blacklevel1,size_new)
    blacklevel2=cv2.resize(blacklevel2,size_new)
    blacklevel3=cv2.resize(blacklevel3,size_new)
    blacklevel4=cv2.resize(blacklevel4,size_new)
    C1 = C1 - blacklevel1
    C2 = C2 - blacklevel2
    C3 = C3 - blacklevel3
    C4 = C4 - blacklevel4
    img = raw_image.simple_integration(C1, C2, C3, C4)
    img=np.clip(img,a_min=0,a_max=np.max(img))
    return img
if __name__ == "__main__":
    print ('This is main of BLC')
    img=plained_raw.read_plained_file("DSC16_1339_768x512_rggb.raw",height=512,width=768,shift_bits=0)
    #img2=blc_process(img,38,38,38,38)
    blacklevel=np.ones((32,48))*38
    img3=block_blc(img,blacklevel,blacklevel,blacklevel,blacklevel)
    #plained_raw.write_plained_file("DSC16_1339_768x512_rggb_blc.raw",image=img1)