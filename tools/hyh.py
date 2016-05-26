#!/usr/bin/env python

# --------------------------------------------------------
# 
# Copyright (c) 2015
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------

"""Test a region proposal network on an image database."""

import _init_paths
from rpn.test import test_net
from rpn.test import im_detect
from rpn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
import datasets
import caffe
import argparse
import pprint
import time, os, sys, cv2
from utils.cython_nms import nms
from utils.timer import Timer
from utils.cython_bbox import bbox_overlaps
import matplotlib.pyplot as plt
import numpy as np
import cPickle
from utils import exif
from pymongo import MongoClient, ReturnDocument
import re
import math
# configuration
CONF_THRESH = 0.5
NMS_CONFIG = {'USE_GPU': False,
              'NMS_THRESH': 0.3,
              'PRE_NMS_TOPN': 2000,
              'POST_NMS_TOPN': 1}
EXPAND_RATIO = 1/10.0
# expand_val = lambda boxes: np.array([boxes[:,0] - boxes[:,2], boxes[:,1] - boxes[:,3],
#                                      boxes[:,2] - boxes[:,0], boxes[:,3] - boxes[:,1],
#                                      np.zeros(boxes.shape[0])]).T * EXPAND_RATIO    


def demo(net, im_path, anchor_file, des_dir='demo'):
    # Load the demo image
    if not os.path.exists(im_path):
        print 'Image `{:s}` not found!'.format(im_path)
        return 
    #im = cv2.imread(im_path)
    im = exif.load_exif_jpg(im_path)
    #wdth = 1280
    #eight = 720
    #ulti = img.shape[0]*img.shape[1]*1.0/(width*height)


    
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, anchor_file)
    #print scores
    #print boxes
    #print len(scores)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    cls = 'obj'
    dets = np.hstack((boxes, scores)).astype(np.float32, copy=False)
    dets = boxes_filter(dets, NMS_CONFIG['PRE_NMS_TOPN'], 
                              NMS_CONFIG['NMS_THRESH'], 
                              NMS_CONFIG['POST_NMS_TOPN'],
                              CONF_THRESH
                        )
    print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls, CONF_THRESH)
    #vis_detections(im, cls, dets, thresh=CONF_THRESH)
    #print dets
    # save result images
    output_dir = os.path.join(cfg.ROOT_DIR, 'data', des_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # image_name = os.path.splitext(os.path.basename(im_path))[0]
    # res_im_file = os.path.join(output_dir, 'res_'+image_name+'.jpg')
    image_name = '_'.join(im_path.split('/')[-3:])
    res_im_file = os.path.join(output_dir, image_name)

    #save_detection_res(im, res_im_file, dets)
    return (dets)


def boxes_filter(dets, PRE_NMS_TOPN, NMS_THRESH, POST_NMS_TOPN, 
                 CONF_THRESH, USE_GPU=False):
    """ filter the proposal boxes """
    # speed up nms 
    if PRE_NMS_TOPN > 0:
        dets = dets[: min(len(dets), PRE_NMS_TOPN), :]

    # apply nms
    if NMS_THRESH > 0 and NMS_THRESH < 1:
        if USE_GPU:
            keep = nms_gpu(dets, NMS_THRESH)
        else:
            keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

    if POST_NMS_TOPN > 0:
        dets = dets[: min(len(dets), POST_NMS_TOPN), :]

    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    dets = dets[inds, :] 

    return dets


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i, det in enumerate(dets):
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

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def save_detection_res(im, path, dets, gt_boxes=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    boxes_size = []
    
    if gt_boxes != None:
        for gt_box in gt_boxes:
            cv2.rectangle(im, (int(gt_box[0]), int(gt_box[1])),
                              (int(gt_box[2]), int(gt_box[3])),
                              (0, 0, 255), 2)

    for i in xrange(len(dets)):
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[2]), int(bbox[3])), 
                    (0, 255, 0), 2)
        cv2.putText(im, str(i), (int(bbox[0]), int(bbox[1])), font, 0.5, (0,0,255), 1)
        boxes_size.append([int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])])
    
    cv2.rectangle(im, (0,0), (300, (len(boxes_size)+1)*20), (125,125,125), -1)
    for i in xrange(len(dets)):
        text_content = ('{:d}.scores: {:.3f}, box sixe: [{:d},{:d}]').format(i, 
                                dets[i, -1], boxes_size[i][0], boxes_size[i][1])
        cv2.putText(im, text_content, (0, (i+1)*20), font, 0.5, (255,0,0), 1)
    
    cv2.imwrite(path, im)


def test_imdb(net, imdb, anchors):
    """ Test a region proposal network on a image dataset  """
    output_dir = get_output_dir(imdb, net)
    cache_file = os.path.join(output_dir, 'res_boxes.pkl')
    
    # load cache result boxes (filtered)
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            proposal_boxes = cPickle.load(f)
        print 'load res boxes from \'{}\''.format(cache_file)
        return proposal_boxes
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    print 'Generating proposal boxes by rpn model...'
    proposal_boxes = test_net(net, imdb, anchors)
    print 'Get proposal boxes done!'
    
    print 'Current NMS configuration:'
    print NMS_CONFIG

    expand_val = lambda boxes: np.array([boxes[:,0] - boxes[:,2], boxes[:,1] - boxes[:,3],
                                     boxes[:,2] - boxes[:,0], boxes[:,3] - boxes[:,1],
                                     np.zeros(boxes.shape[0])]).T * EXPAND_RATIO    
    
    # filter boxes
    print 'Filtering proposal boxes...'
    for i in xrange(len(proposal_boxes)):
        proposal_boxes[i] = boxes_filter(proposal_boxes[i], 
                PRE_NMS_TOPN=NMS_CONFIG['PRE_NMS_TOPN'], 
                NMS_THRESH=NMS_CONFIG['NMS_THRESH'], 
                POST_NMS_TOPN=NMS_CONFIG['POST_NMS_TOPN'],
                CONF_THRESH=CONF_THRESH,
                USE_GPU=NMS_CONFIG['USE_GPU'])

        # expand bounding box
        if len(proposal_boxes[i]) > 0:
            proposal_boxes[i] = proposal_boxes[i] + expand_val(proposal_boxes[i])
        print 'filter proposal box: {:d}/{:d}'.format(i+1, len(proposal_boxes))
    print 'Filter proposal boxes done!'
    
    # save file
    with open(cache_file, 'wb') as f:
        cPickle.dump(proposal_boxes, f, cPickle.HIGHEST_PROTOCOL)
        print 'save result boxes to `{:s}`'.format(cache_file)
 
    return proposal_boxes


def calc_precision_recall(all_boxes, imdb):
    res_num = {'tp': 0, 'gt': 0, 'det': 0, 'bad_case': 0}
    
    # save bad case result 
    bad_case_output_dir = os.path.join(cfg.ROOT_DIR, 'data', 'bad_case_'+imdb.name)
    if not os.path.exists(bad_case_output_dir):
        os.makedirs(bad_case_output_dir)
    else:
        for f in os.listdir(bad_case_output_dir):
            os.remove(os.path.join(bad_case_output_dir, f))

    gt_roidb = imdb.roidb
    outside_pad = 10
    bounding = lambda box, gt_box: np.all((box[:2] <= gt_box[:2] + outside_pad) & 
                                          (box[2:] >= gt_box[2:] - outside_pad))

    for im_i, boxes in enumerate(all_boxes):
        gt_boxes = gt_roidb[im_i]['boxes']
        gt_overlaps = bbox_overlaps(boxes[:,:-1].astype(np.float), 
                                    gt_boxes.astype(np.float))
        argmaxes = gt_overlaps.argmax(axis=1)
        """ 
        maxes = gt_overlaps.max(axis=1)
        tp_inds = np.where(maxes >= 0.7)[0]
        """
        tp_inds = np.zeros((argmaxes.shape[0]), dtype=bool)
        for box_i, box in enumerate(boxes):
            if bounding(box[:-1], gt_boxes[argmaxes[box_i]]):
                tp_inds[box_i] = True

        tp_argmaxes = argmaxes[tp_inds]
        tp_argmaxes = np.unique(tp_argmaxes)
        tp_num = tp_argmaxes.size
        
        res_num['tp'] = res_num['tp'] + tp_num
        res_num['gt'] = res_num['gt'] + len(gt_boxes)
        res_num['det'] = res_num['det'] + len(boxes)

        if tp_num != len(boxes) or tp_num != len(gt_boxes):
            res_num['bad_case'] = res_num['bad_case'] + 1
            img_path = imdb.image_path_at(im_i)
            im = cv2.imread(img_path)
            bad_name = os.path.splitext(os.path.basename(img_path))[0]
            res_im_file = os.path.join(bad_case_output_dir, '{:s}.jpg'.format(bad_name))
            save_detection_res(im, res_im_file, boxes, gt_boxes)
            print 'images: {:d}/{:d}  !!!  BAD CASE'.format(im_i, len(all_boxes))
        else:
            print 'images: {:d}/{:d}'.format(im_i, len(all_boxes))

    print '=' * 20
    print 'final bad case number: {:d}'.format(res_num['bad_case'])
    print 'final precision: {:.3f}, recall: {:.3f}.'.format(
                                        float(res_num['tp'])/float(res_num['det']), 
                                        float(res_num['tp'])/float(res_num['gt']))
    print '=' * 20    

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a region proposal network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--defining', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--anchors', dest='anchors',
                        help='base anchor boxes',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default=None, type=str)
    parser.add_argument('--bad_list', dest='bad_list', help='bad list file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


class RCNNDetector:
    def __init__(self):
        #, defining, net, anchors, cfg):
        self.defining ='models/RPN_Net/ZF/test.prototxt'
        #self.net ='models/rpn_zf_mix_overlap_09_plate_gz_10w_trainval.caffemodel' 
        #self.anchors = 'data/cache/plate_phone_all_mix_overlap_09_anchors.pkl'
        self.cfg ='/home/yaoliu/ObjectDetection/plate-detection/faster-rcnn/experiments/cfgs/mix_overlap_09.yml'
        self.net='output/mix_overlap_09/plate_database_all/rpn_zf_mix_overlap_09_plate_database_all.caffemodel'
        self.anchors='data/cache/plate_database_all_mix_overlap_09_anchors.pkl'
    def predict(self):
        #args = parse_args() 
        prototxt = self.defining
        caffemodel = self.net
        anchors = self.anchors

        #print('Called with args:')
        #print(args)

        
        cfg_from_file(self.cfg)

        print('Using config:')
        pprint.pprint(cfg)

        while not os.path.exists(caffemodel):
            print('Waiting for {} to exist...'.format(caffemodel))
            time.sleep(10)
        
    
        caffe.set_mode_gpu()
        caffe.set_device(0)

        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(caffemodel))[0]
        print '\n\nLoaded network {:s}'.format(caffemodel)
        
       
        #img_list = ['tr.jpg']
        #for img_name in img_list:
         #   print '~' * 20
          #  im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', img_name)
           # print 'Demo for image: `{:s}`'.format(im_file)
           # demo(net, im_file, anchors)

           # plt.show()
        db = MongoClient('10.76.0.143', 27017)['plate']
        db.authenticate('plateReader', 'IamReader')
        total_num=db.image.count()
       # total_num=1407
        #bad=open('bad_list.txt','r')
        bad=open('id_path.txt','r')
        #bad=open('id_path.txt','r')
        #print total_num
        #idlist=range(total_num)
        #print idlist
        error1=0
        f=open('re_predict_result_new.txt','w')
        for idlist in bad:
        #for i in bad:
           # num=int(i)-1
            #print count
            #print num
            #print type(num)
            re_num=re.split(' ',idlist)   
            #print re_num[0]
            num=int(re_num[0])
            #print num
            temp=db.image.find_one({'_id':int(num)})

            #temp=db.image.find_one({'_id',num+1})
            if temp:
                #temp=db.image.find_one({'_id':num+1})
                im_file=temp['path']
                img = exif.load_exif_jpg(im_file)
                width = 1280
                height = 720
                multi = img.shape[0]*img.shape[1]*1.0/(width*height)
                multi = math.sqrt(multi)
                print multi
                resized_img = cv2.resize(img,(int(img.shape[1]/multi), int(img.shape[0]/multi)))
                resized_img_name = os.path.join('plate_buffer', 'resized_img.jpg')
                cv2.imwrite(resized_img_name, resized_img)
                re_img = cv2.imread(resized_img_name)
                #e_img_size = re_img.shape

                print im_file
                try:
                    presicion= demo(net,resized_img_name,anchors)
                    print presicion
                    #print 'result'
                    #print presicion
                    #print int(presicion[0][2:3])
                    #print "a"
                    #print len(temp['points'])
                    #print "b"
                    if len(temp['points'])>0:
                        f.write(str(num)+' ')
                        #f.write(str(temp['points'][0][0][0:1][0])+' '+str(temp['points'][0][0][1:2][0])+' ')
                        #f.write(str(temp['points'][0][1][0:1][0])+' '+str(temp['points'][0][1][1:2][0])+' ')
                        #f.write(str(temp['points'][0][2][0:1][0])+' '+str(temp['points'][0][2][1:2][0])+' ')
                        #f.write(str(temp['points'][0][3][0:1][0])+' '+str(temp['points'][0][3][1:2][0])+' ')
                        #print 'flag1'
                        if len(presicion)!=0:
                            f.write('*'+' ')
                            f.write(str(int(presicion[0][2:3]*multi+1))+' '+str(int(presicion[0][1:2]*multi-1))+' ')
                            f.write(str(int(presicion[0][2:3]*multi+1))+' '+str(int(presicion[0][3:4]*multi+1))+' ')
                            f.write(str(int(presicion[0][0:1]*multi-1))+' '+str(int(presicion[0][3:4]*multi+1))+' ')
                            f.write(str(int(presicion[0][0:1]*multi-1))+' '+str(int(presicion[0][1:2]*multi-1)))
                            #error1=error1+1
                        else:
                            f.write('* '+'0 '+'0 '+'0 '+'0 '+'0 '+'0 '+'0 '+'0')
                        f.write('\n')
                   # print 'flags2'
                except:
                        print 'we are wrong'
                        pass
            #break           
        f.close()
        #f.write(str(int(total_num-error1)))                    
            #f.write(str(presicion[]))
            #print 'result'
            #print int( presicion[0][1:2])
        






detector = RCNNDetector()
detector.predict()
