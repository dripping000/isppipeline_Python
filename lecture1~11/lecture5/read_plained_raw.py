
import numpy as np
import raw_image_show as rawshow
import math

def read_plained_file(file_path_name,height,width,shift_bits):
    frame = np.fromfile(file_path_name, dtype="uint16")
    # print("b shape",b.shape)
    # print('%#x'%b[0])
    frame=frame[0:height*width]
    frame.shape = [height, width]
    frame=np.right_shift(frame, shift_bits)
    return frame

def test_case_read_planed_10():
    file_name="RAW_GRBG_plained_4608(9216)x3456_A.raw"
    image=read_plained_file(file_name,3456,4608,0)
    image=image/1023
    #raw_image_show_thumbnail(image,4608,3456)
    rawshow.raw_image_show_thumbnail(image, 3456, 4608)


if __name__ == "__main__":
    print ('This is main of module')
    x=test_case_read_planed_10()
    print(x)


