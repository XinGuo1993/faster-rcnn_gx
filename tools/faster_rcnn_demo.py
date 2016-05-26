#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015
# Licensed under The MIT License [see LICENSE for details]
# Written by
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from utils.cython_nms import nms
from utils.timer import Timer
from fast_rcnn import test as fast_rcnn_test
from fast_rcnn import config as fast_rcnn_config
from rpn import config as rpn_config
from rpn import test as rpn_test
from datasets.factory import get_imdb
import matplotlib.pyplot as plt
import test_net as test_fast_rcnn_net
import test_rpn_net
import datasets
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import pprint


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
 
NETS = {'vgg16': ('RPN_Net/VGG16',      # rpn test net def-file
                  'rpn_vgg16_fc6_voc_2007_final.caffemodel',  # rpn model
                  'fast_rcnn/VGG16',    # fast-rcnn test net def-file    
                  'fast_rcnn_vgg16_fc6_voc_2007_final.caffemodel',    # fast-rcnn model
                  'voc_2007_trainval_anchors.pkl',       # base anchors
                  'conv5_3'    # last shared layer name
                  ),
        'zf': ('RPN_Net/ZF',
                  'rpn_zf_fc6_voc_2007_final.caffemodel',
                  'fast_rcnn/ZF',
                  'fast_rcnn_zf_fc6_voc_2007_final.caffemodel',
                  'voc_2007_trainval_anchors.pkl',
                  'conv5'
                  ),
        'zf_ren': ('ZF_Ren/rpn',
                  'proposal_final',
                  'ZF_Ren/fast_rcnn',
                  'detection_final',
                  'voc_2007_trainval_anchors.pkl',
                  'conv5'
                  ),
        'vgg16_ren': ('ZF_Ren/rpn_vgg16',
                  'vgg16_proposal_final',
                  'ZF_Ren/fast_rcnn_vgg16',
                  'vgg16_detection_final',
                  'voc_2007_trainval_anchors.pkl',
                  'conv5_3'
                  ),
        }


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    ax = plt.axes()
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(rpn_net, fast_rcnn_net, anchor_file, image_name, classes=CLASSES, 
        share_conv=False, last_shared_blob_name=None):
    """
    Detect object classes in an image.
    The proposals are generating by region proposal network.
    """
    # Load the demo image
    im_file = os.path.join(rpn_config.cfg.ROOT_DIR, 'data', 'demo', image_name)
    im = cv2.imread(im_file)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    
    # ####### RPN 
    # Generate the proposal boxes
    _t['im_detect'].tic()
    scores, boxes = rpn_test.im_detect(rpn_net, im, anchor_file)
    _t['im_detect'].toc()
    print 'image: {:s} num proposal: {:d}'.format(image_name, boxes.shape[0])
    
    # Filter the proposal boxes
    _t['misc'].tic()
    obj_proposals = np.hstack((boxes, scores)).astype(np.float32, copy=False)
    obj_proposals = test_rpn_net.boxes_filter(obj_proposals, 6000, 0.7, 300)
    _t['misc'].toc()
    print 'image: {:s} num proposal filtered: {:d}'.format(image_name, 
                                                    obj_proposals.shape[0])

    print ('Actions took {:.3f}s for generating'
            '{:d} proposal boxes, {:.3f}s for '
            'filtering proposals.').format(_t['im_detect'].total_time, 
                    boxes.shape[0], _t['misc'].total_time)


    # ###### Fast-RCNN
    # Detect all object classes and regress object bounds
    _t['im_detect'].tic()
    if share_conv:
        # conv_feat_blob = rpn_net.blobs['conv5'].data
        scores, boxes = fast_rcnn_test.im_detect(fast_rcnn_net, im, obj_proposals[:, 0:4],
                                            rpn_net.blobs[last_shared_blob_name].data)
    else:
        scores, boxes = fast_rcnn_test.im_detect(fast_rcnn_net, im, obj_proposals[:, 0:4])
    _t['im_detect'].toc()
    print 'image: {:s} num obj boxes: {:d}'.format(image_name, boxes.shape[0])
    
    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.3
    _t['misc'].tic()

    plt.figure()
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls, CONF_THRESH)
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    _t['misc'].toc()
    
    print 'Actions took {:.3f}s for detecton, {:.3f}s for nms.' \
              .format( _t['im_detect'].total_time, _t['misc'].total_time)


def test_imdb(rpn_net, fast_rcnn_net, imdb, anchors):
    """ Test Faster-rcnn model on a image dataset  """
    print 'Run RPN model, get the proposalboxes...'
    proposal_boxes = test_rpn_net.test_imdb(rpn_net, imdb, anchors)
    print 'done!' 

    print 'Append proposal boxes into imdb'
    roidb = []
    for box in proposal_boxes:
        roidb.append({'boxes': box[:, 0:4], 
                      'gt_classes' : np.zeros((len(box),),
                                    dtype=np.int32)})
    imdb.roidb = roidb
    print 'done!'

    print 'Run Fast-RCNN model to detect...'
    fast_rcnn_test.test_net(fast_rcnn_net, imdb)
    print 'done!'


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--share', dest='share_conv',
                        help='Share conv layers ()',
                        action='store_true')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    print('Called with args:')
    print(args)
    
    rpn_prototxt = os.path.join(rpn_config.cfg.ROOT_DIR, 'models', 
                                    NETS[args.demo_net][0], 'test.prototxt')
    rpn_caffemodel = os.path.join(rpn_config.cfg.ROOT_DIR, 'data', 'rpn_models',
                                    NETS[args.demo_net][1])
    if args.share_conv and not args.imdb_name:
        fast_rcnn_prototxt = os.path.join(rpn_config.cfg.ROOT_DIR, 'models', 
                                    NETS[args.demo_net][2], 'test_share.prototxt')
        last_shared_blob_name = NETS[args.demo_net][5]
    else:
        fast_rcnn_prototxt = os.path.join(rpn_config.cfg.ROOT_DIR, 'models', 
                                    NETS[args.demo_net][2], 'test.prototxt')
        last_shared_blob_name = None
    fast_rcnn_caffemodel = os.path.join(rpn_config.cfg.ROOT_DIR, 'data', 
                                    'fast_rcnn_models', NETS[args.demo_net][3])
    anchors = os.path.join(rpn_config.cfg.ROOT_DIR, 'data', 'rpn_models', 
                                    NETS[args.demo_net][4])
    
    if not os.path.isfile(rpn_caffemodel):
        raise IOError(('{:s} not found.').format(rpn_caffemodel))
    if not os.path.isfile(fast_rcnn_caffemodel):
        raise IOError(('{:s} not found.').format(fast_rcnn_caffemodel))
    
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)

    rpn_net = caffe.Net(rpn_prototxt, rpn_caffemodel, caffe.TEST)
    rpn_net.name = os.path.splitext(os.path.basename(rpn_caffemodel))[0]
    print '\n\nLoaded region proposal network {:s}'.format(rpn_caffemodel)

    fast_rcnn_net = caffe.Net(fast_rcnn_prototxt, fast_rcnn_caffemodel, caffe.TEST)
    fast_rcnn_net.name = os.path.splitext(os.path.basename(fast_rcnn_caffemodel))[0]
    print '\n\nLoaded fast-rcnn network {:s}'.format(fast_rcnn_caffemodel)
     
    # Do detection
    if args.imdb_name:  # test on a dataset
        imdb = get_imdb(args.imdb_name)
        test_imdb(rpn_net, fast_rcnn_net, imdb, anchors)
    else:   # test demo
        img_list = ['000004.jpg', '004545.jpg', '000012.jpg', '000142.jpg']
        classes = ('car', 'person', 'dog', 'horse')
        
        for img_name in img_list:
            print '~' * 20
            print 'Demo for image: data/demo/' + img_name
            demo(rpn_net, fast_rcnn_net, anchors, img_name, classes,
                args.share_conv, last_shared_blob_name)

        plt.show()
