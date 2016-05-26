#!/usr/bin/python
#

import sys
import PIL
from PIL import Image
import numpy as np
import cv2

exif_orientation_table = [{'angle': 0, 'flip': False},  # 0-ingore
                          {'angle': 0, 'flip': False},  # 1-horizontal(normal)
                          {'angle': 0, 'flip': True},   # 2-mirror horizontal
                          {'angle': 180, 'flip': False},# 3-rotate 180
                          {'angle': 180, 'flip': True}, # 4-mirror veritical
                          {'angle': 90, 'flip': True},  # 5-mirror horizontal and rotate 270 CW
                          {'angle': 270, 'flip': False},# 6-rotate 60 CW
                          {'angle': 270, 'flip': True}, # 7-mirror horizontal and rotate 90 CW
                          {'angle': 90, 'flip': False}  # 8-rotate 270 CW
                         ]

def rotate_exif_img(src_img, rotate_flag):
    # ingore index out of range
    if rotate_flag >= len(exif_orientation_table) or rotate_flag < 0:
        rotate_flag = 0

    rotate_angle = exif_orientation_table[rotate_flag]['angle']
    flip_flag = exif_orientation_table[rotate_flag]['flip']

    if rotate_angle != 0:
        dest_img = np.rot90(src_img, rotate_angle/90)
        dest_img = dest_img.copy()
    else:
        dest_img = src_img

    if flip_flag:
        dest_img = dest_img[:, ::-1, :]

    return dest_img


def load_exif_jpg(img_path):
    pil_im = PIL.Image.open(img_path)
    flag_key = 274

    try:
        exif_data = pil_im._getexif()
        if exif_data and flag_key in exif_data.keys():
            # print 'orientation:', exif_data[flag_key]
            orientation_flag = int(exif_data[flag_key])
        else:
            orientation_flag = 0
    except ZeroDivisionError:
        # bug: Pillow==3.0
        print 'error'
        orientation_flag = 0
    except:
        # it is not a jpg format image
        orientation_flag = 0
    
    # convert to opencv format (RBG to BGR)
    opencv_img = np.array(pil_im)[:, :, ::-1].copy()
    final_img = rotate_exif_img(opencv_img, orientation_flag)

    return final_img


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) != 2:
        print 'usage:\n\t{:s} jpg_img'.format(argv[0])
        sys.exit()
    
    # raw
    im_raw = cv2.imread(argv[1])
    resize_img = cv2.resize(im_raw, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite('raw_resized.jpg', resize_img)

    # rotate
    final_img = load_exif_jpg(argv[1])
    cv2.imwrite('exif_res.jpg', final_img)
    
    # resize
    resize_img = cv2.resize(final_img, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite('exif_resized.jpg', resize_img)
    
    # draw rectangle
    cv2.rectangle(final_img, (100, 100), (200, 200), (0, 255, 0), 2)
    cv2.imshow('test', final_img)
    cv2.waitKey(0)
