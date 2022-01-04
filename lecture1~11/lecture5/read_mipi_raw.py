
import numpy as np
import raw_image_show as rawshow
import math

def read_mipi10_file(file_path_name,height,width):

    new_width = int(math.floor((width + 3) / 4) * 4)
    packet_num_L = int(new_width / 4)
    width_byte_num = packet_num_L * 5
    width_byte_num = int(math.floor((width_byte_num + 7) / 8) * 8)
    image_bytes=width_byte_num*height
    frame = np.fromfile(file_path_name, count=image_bytes,dtype ="uint8")
    print("b shape",frame.shape)
    print('%#x'%frame[0])
    frame.shape = [height, width_byte_num]
    one_byte = frame[:,0:image_bytes:5]
    two_byte = frame[:,1:image_bytes:5]
    three_byte = frame[:,2:image_bytes:5]
    four_byte = frame[:,3:image_bytes:5]
    five_byte = frame[:,4:image_bytes:5]
    one_byte=one_byte.astype('uint16')
    two_byte= two_byte.astype('uint16')
    three_byte = three_byte.astype('uint16')
    four_byte= four_byte.astype('uint16')
    five_byte= five_byte.astype('uint16')

    one_byte = np.left_shift(one_byte, 2) + np.bitwise_and((five_byte), 3)
    two_byte = np.left_shift(two_byte, 2) + np.right_shift(np.bitwise_and((five_byte), 12), 2)
    three_byte = np.left_shift(three_byte, 2) + np.right_shift(np.bitwise_and((five_byte), 48), 4)
    four_byte = np.left_shift(four_byte, 2) + np.right_shift(np.bitwise_and((five_byte), 192), 6)

    frame_pixels=np.zeros(shape=(height,new_width))
    frame_pixels[:, 0: new_width:4]=one_byte[:, 0: packet_num_L]
    frame_pixels[:, 1: new_width:4]=two_byte[:, 0: packet_num_L]
    frame_pixels[:, 2: new_width:4]=three_byte[:, 0: packet_num_L]
    frame_pixels[:, 3: new_width:4]=four_byte[:, 0: packet_num_L]
    frame_out=frame_pixels[:,0:width]
    return frame_out

def test_case_read_mipi_10():
    file_name="original_mipi10_4032X3016.raw"
    done_file_name =  "done" + file_name
    image=read_mipi10_file(file_name,3016, 4032)
    image=image/1023
    #raw_image_show_thumbnail(image,4608,3456)
    rawshow.raw_image_show_thumbnail(image, 3016, 4032)





if __name__ == "__main__":
    print ('This is main of module')
    x=test_case_read_mipi_10()
    print(x)