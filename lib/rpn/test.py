# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from rpn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.cython_nms import nms
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os, sys

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)

        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors


def _bbox_pred(boxes, box_deltas):
    """Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
    heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:  , 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _filter_boxes(min_box_size, boxes, scores):
    """ To filter small boxes """
    widths = boxes[:, 2] - boxes[:, 0] + 1
    heights = boxes[:, 3] - boxes[:, 1] + 1

    valid_inds = (widths >= min_box_size) & (heights >= min_box_size)

    boxes = boxes[valid_inds]
    scores = scores[valid_inds]
    
    return boxes, scores


def im_detect(net, im, base_anchors_file):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): RPN network to use
        im (ndarray): color image to test (in BGR order)
    Returns:
        scores (ndarray): [H*W*k] x 2 array of object class scores (k-anchors)
        boxes (ndarray): [H*W*k] x 4 array of predicted bounding boxes
    """
    if os.path.exists(base_anchors_file):
        with open(base_anchors_file, 'rb') as f:
            base_anchors = cPickle.load(f)
    else:
        print 'Can not load base anchors from {}.'.format(base_anchors_file)
        sys.exit()

    timers = {'t_pre': Timer(), 't_forward': Timer(), 't_post': Timer()}
    timers['t_pre'].tic()
    
    blobs, im_scale = _get_blobs(im)

    #rint im_scale

    #print 'asdsaddsadaassdsdsdsaadsadsa'
    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    
    timers['t_pre'].toc()
    timers['t_forward'].tic()
    
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False))

    timers['t_forward'].toc()
    timers['t_post'].tic()

    # get anchor boxes
    proposal_cls_score = net.blobs['proposal_cls_score'].data
    featuremap_size = [np.size(proposal_cls_score, 2), np.size(proposal_cls_score, 3)]
    anchors = proposal_locate_anchors_with_featuremap(base_anchors, featuremap_size)    

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['proposal_cls_score'].data[:,1::2,:,:]
    else:
        # use softmax estimated probabilities
        scores = blobs_out['proposal_cls_prob'][:,1:,:,:]
        scores = scores.reshape((scores.shape[0], -1, proposal_cls_score.shape[2], 
                                    proposal_cls_score.shape[3]))
 
    # reshape cls_score result dimension
    scores = np.transpose(scores, (0, 3, 2, 1))
    scores = scores.reshape((-1, 1))

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['proposal_bbox_pred']
        
        # reshape bbox_pred result dimeinsion
        box_deltas = np.transpose(box_deltas, (0, 3, 2, 1))
        box_deltas = box_deltas.reshape((-1, 4))

        pred_boxes = _bbox_pred(anchors, box_deltas)
        # sclae back
        raw_size = np.array([im.shape[1], im.shape[0], im.shape[1], im.shape[0]]) - 1
        scaled_im_size = im.shape * im_scale
        scaled_size = np.array([scaled_im_size[1], scaled_im_size[0], 
                                scaled_im_size[1], scaled_im_size[0]]) - 1
        scale_factor = raw_size / scaled_size
        pred_boxes = pred_boxes * scale_factor

        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        # pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        pred_boxes = anchors

    if cfg.ANCHOR.TEST_DROP_BOXES_RUNOFF_IMG:
        im_size_scaled = np.round(im.shape[0:2] * im_scale)
        contained_in_image = np.all((anchors >= 0) & 
                (anchors <= np.array([im_size_scaled[1], im_size_scaled[0], 
                im_size_scaled[1], im_size_scaled[0]])), axis=1)
        scores = scores[contained_in_image, :]
        pred_boxes = pred_boxes[contained_in_image]

    # drop too small boxes
    pred_boxes, scores = _filter_boxes(cfg.TEST.MIN_BOX_SIZE, pred_boxes, scores)

    # sort the boxes
    sorted_inds = scores[:,0].argsort()[::-1]
    pred_boxes = pred_boxes[sorted_inds]
    scores = scores[sorted_inds]

    timers['t_post'].toc()
    print 'pre: {:.3f}s forword: {:.3f}s post: {:.3f}s'.format(timers['t_pre'].total_time,
                            timers['t_forward'].total_time, timers['t_post'].total_time)

    return scores, pred_boxes


def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net(net, imdb, base_anchors_file):
    """ Test a region proposal network on a image database. """
    output_dir = get_output_dir(imdb, net)
    cache_file = os.path.join(output_dir, 'proposal_boxes.pkl')
    
    # load cache proposal boxes
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            proposal_boxes = cPickle.load(f)
        print 'load proposal boxes from \'{}\''.format(cache_file)
        return proposal_boxes
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #######  have no cache proposal boxes
    num_images = len(imdb.image_index)

    # all detections are collected into:
    #    all_boxes[image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[] for _ in xrange(num_images)]
    
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    
    # generate proposal boxes
    for i in xrange(num_images):
        img_path = imdb.image_path_at(i)
        _t['im_detect'].tic()
        im = cv2.imread(img_path)
        scores, boxes = im_detect(net, im, base_anchors_file)
        _t['im_detect'].toc()
        
        all_boxes[i] = np.hstack((boxes, scores)).astype(np.float32, copy=False)
        
        print 'gen_proposal: {:d}/{:d} {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time)
    
    # save file
    with open(cache_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
    
    return all_boxes


#####################################
def proposal_locate_anchors_with_featuremap(base_anchors, featuremap_size):
    shift_x = np.array([ i/cfg.DEDUP_BOXES for i in range(0, featuremap_size[1]) ])
    shift_y = np.array([ i/cfg.DEDUP_BOXES for i in range(0, featuremap_size[0]) ])
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    
    # obtain all anchor boxes
    shift_x_y = np.array([shift_x.flatten('F'), shift_y.flatten('F'), shift_x.flatten('F'), shift_y.flatten('F')]).T
    # final_anchors = np.repeat(base_anchors, shift_x_y.shape[0], axis=0) + np.tile(shift_x_y, (base_anchors.shape[0], 1))
    final_anchors = np.tile(base_anchors, (shift_x_y.shape[0], 1)) + np.repeat(shift_x_y, base_anchors.shape[0], axis=0)

    return final_anchors
