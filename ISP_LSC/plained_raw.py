
import numpy as np
import raw_image_show as rawshow
import math

def read_plained_file(file_path_name, height, width, shift_bits):
    frame = np.fromfile(file_path_name, dtype="uint16")
    # print('%#x' % frame[0])

    frame = frame[0:height*width]
    frame.shape = [height, width]
    frame = np.right_shift(frame, shift_bits)

    # print("shape", frame.shape)

    return frame

def write_plained_file(file_path_name,image):
    image=image.astype(np.uint16)
    frame = image.tofile(file_path_name)
    return

def test_case_read_planed_10():
    file_name="RAW_GRBG_plained_4608(9216)x3456_A.raw"
    image=read_plained_file(file_name,3456,4608,0)
    #raw_image_show_thumbnail(image,4608,3456)
    rawshow.raw_image_show_thumbnail(image/1023, 3456, 4608)
    write_plained_file("test_result.raw",image)

if __name__ == "__main__":
    print ('This is main of module')
    x=test_case_read_planed_10()
    print(x)


