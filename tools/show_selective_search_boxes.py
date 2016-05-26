#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# 
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
import caffe, os, sys, cv2
import scipy.io as sio
import argparse



def vis_selective_serach_boxes(image_name, image_type='jpg'):
    # Load pre-computed Selected Search object proposals
    box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
                            image_name + '_boxes.mat')
    obj_proposals = sio.loadmat(box_file)['boxes']
    
    print 'proposals numbers: ' + str(len(obj_proposals))

    # Load the demo image
    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name + '.' + image_type)
    im = cv2.imread(im_file)
    
    result_base_path = 'cache/'

    # print obj_proposals[1, :]

    for ix, obj_proposal in enumerate(obj_proposals):
        # sub_image = im[obj_proposal[0]-1: obj_proposal[2], obj_proposal[1]-1: obj_proposal[3]]
        sub_image = im[obj_proposal[1]: obj_proposal[3]+1, obj_proposal[0]: obj_proposal[2]+1]
        dest_image_path = result_base_path + str(ix) + '.png'
        cv2.imwrite(dest_image_path, sub_image)

    
    print 'done'
    # cv2.imshow('test show', im)
    # cv2.waitKey(0)    


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Show Proposals of a image')
    parser.add_argument('--image', dest='image_name', help='test image name',
                        default='000004')
    parser.add_argument('--type', dest='image_type', help='test image type',
                        default='jpg')


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'SS for data/demo/' + args.image_name + '.' + args.image_type
    vis_selective_serach_boxes(args.image_name, args.image_type)


