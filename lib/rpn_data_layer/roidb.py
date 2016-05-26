# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from rpn.config import cfg
import utils.cython_bbox
from utils.blob import prep_im_for_blob
import cv2


def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

    # debug code
    # import os, cPickle, rpn
    # cache_file = os.path.join(rpn.ROOT_DIR, 'data', 'cache', 'test_new_roidb.pkl')
    # with open(cache_file, 'wb') as f:
    #     cPickle.dump(roidb, f, cPickle.HIGHEST_PROTOCOL)
    # print 'write test roidb file...'
        
    
def add_bbox_regression_targets(roidb, anchordb):
    """Add information needed to train bounding-box regressors."""
    # assert len(roidb) > 0
    # assert 'max_classes' in roidb[0], 'Did you call prepare_roidb first?'

    num_images = len(roidb)
    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    for im_i in xrange(num_images):
        im = cv2.imread(roidb[im_i]['image'])
        anchors, im_scales = proposal_locate_anchors(im, anchordb)
       
        rois = roidb[im_i]['boxes']
        # max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        im_size = np.array(im.shape[0:2])
        roidb[im_i]['bbox_targets'] = []
        roidb[im_i]['max_overlaps'] = []
       
        # multi-scale bbox_target
        for ind, im_scale in enumerate(im_scales):
            # roidb[im_i]['bbox_targets'].append( _compute_targets(scale_rois(rois, im_size, im_scale), max_classes, anchors[ind], np.round(im_size * im_scale)) )
            bbox_targets_tmp, max_overlaps_tmp = _compute_targets(
                                                scale_rois(rois, im_size, im_scale), 
                                                max_classes, anchors[ind], 
                                                np.round(im_size * im_scale))
            roidb[im_i]['bbox_targets'].append(bbox_targets_tmp)
            roidb[im_i]['max_overlaps'].append(max_overlaps_tmp)
        
    # Compute values needed for means and stds
    # var(x) = E(x^2) - E(x)^2
    # class_counts = np.zeros((num_classes, 1)) + cfg.EPS
    # sums = np.zeros((num_classes, 4))
    # squared_sums = np.zeros((num_classes, 4))
    class_counts = np.zeros((1, 1)) + cfg.EPS
    sums = np.zeros((1, 4))
    squared_sums = np.zeros((1, 4))

    for im_i in xrange(num_images):
        for scale_i in xrange(len(cfg.TRAIN.SCALES)):
            targets = roidb[im_i]['bbox_targets'][scale_i]
            for cls in xrange(1, num_classes):
                # cls_inds = np.where(targets[:, 0] == cls)[0]
                # if cls_inds.size > 0:
                #     class_counts[cls] += cls_inds.size
                #     sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
                #     squared_sums[cls, :] += (targets[cls_inds, 1:] ** 2).sum(axis=0)
                fg_inds = np.where(targets[:, 0] > 0)[0]
                if fg_inds.size > 0:
                    class_counts[0] += fg_inds.size
                    sums[0, :] += targets[fg_inds, 1:].sum(axis=0)
                    squared_sums[0, :] += (targets[fg_inds, 1:] ** 2).sum(axis=0)

    means = sums / class_counts
    stds = np.sqrt(squared_sums / class_counts - means ** 2)

    # Normalize targets
    for im_i in xrange(num_images):
        for scale_i in xrange(len(cfg.TRAIN.SCALES)):
            targets = roidb[im_i]['bbox_targets'][scale_i]
            # for cls in xrange(1, num_classes):
            #     cls_inds = np.where(targets[:, 0] == cls)[0]
            #     roidb[im_i]['bbox_targets'][scale_i][cls_inds, 1:] -= means[cls, :]
            #     roidb[im_i]['bbox_targets'][scale_i][cls_inds, 1:] /= stds[cls, :]
            fg_inds = np.where(targets[:, 0] > 0)[0]
            if fg_inds.size > 0:
                roidb[im_i]['bbox_targets'][scale_i][fg_inds, 1:] -= means[0, :]
                roidb[im_i]['bbox_targets'][scale_i][fg_inds, 1:] /= stds[0, :]

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel()

def _compute_targets(rois, labels, ex_anchor_rois, im_size_scaled):
    """ Compute bounding-box regression targets for an image. """
    if len(rois) == 0:
        targets = np.zeros((ex_anchor_rois.shape[0], 5), dtype=np.float32)
        max_overlaps = np.zeros((ex_anchor_rois.shape[0], 1), dtype=np.float32)
        targets[:, 0] = -1
        return targets, max_overlaps

    # Ensure gt_labels is in single
    labels = labels.astype(np.float, copy=False)
    assert(np.all(labels > 0))

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = utils.cython_bbox.bbox_overlaps(ex_anchor_rois, rois)

    # drop anchors which run out off image boundaries, if necessary
    if cfg.ANCHOR.DROP_BOXES_RUNOFF_IMG:
        # im_size_scaled = np.round(im_size * im_scale)
        contained_in_image = np.all((ex_anchor_rois >= 0) & (ex_anchor_rois <= np.array([im_size_scaled[1], im_size_scaled[0], im_size_scaled[1], im_size_scaled[0]])), axis=1)
        ex_gt_overlaps[np.logical_not(contained_in_image), :] = 0

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    ex_max_overlaps = ex_gt_overlaps.max(axis=1)
    ex_assignment = ex_gt_overlaps.argmax(axis=1)
    
    # for each gt_rois, get its max overlap with all ex_rois(anchors), the
    # ex_rois(anchors) are recorded in gt_assignment
    # gt_assignment will be assigned as positive 
    # (assign a rois for each gt at least)
    gt_max_overlaps = ex_gt_overlaps.max(axis=0)

    # ex_rois(anchors) with gt_max_overlaps maybe more than one, find them as (gt_best_matches)
    gt_best_matches = np.where(ex_gt_overlaps == gt_max_overlaps)[0]
    
    # Indices of examples for which we try to make predictions
    # both (ex_max_overlaps >= conf.fg_thresh) and gt_best_matches are
    # assigned as positive examples 
    fg_inds = np.unique(np.hstack((np.where(ex_max_overlaps >= cfg.TRAIN.FG_THRESH)[0], gt_best_matches)))
    # fg_inds = np.where(ex_max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    
    # Indices of examples for which we try to used as negtive samples
    # the logic for assigning labels to anchors can be satisfied by both the positive label and the negative label
    # When this happens, the code gives the positive label precedence to
    # pursue high recall
    bg_inds = np.setdiff1d(np.where((ex_max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (ex_max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0], fg_inds)

    if cfg.ANCHOR.DROP_BOXES_RUNOFF_IMG:
       contained_in_image = np.where(contained_in_image)[0]
       fg_inds = np.intersect1d(fg_inds, contained_in_image)
       bg_inds = np.intersect1d(bg_inds, contained_in_image)

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_rois = rois[ex_assignment[fg_inds], :]
    ex_rois = ex_anchor_rois[fg_inds]

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + cfg.EPS
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + cfg.EPS
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + cfg.EPS
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + cfg.EPS
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
    
    # assign the targets
    targets = np.zeros((ex_anchor_rois.shape[0], 5), dtype=np.float32)
    targets[fg_inds, 0] = labels[ex_assignment[fg_inds]]
    targets[fg_inds, 1] = targets_dx
    targets[fg_inds, 2] = targets_dy
    targets[fg_inds, 3] = targets_dw
    targets[fg_inds, 4] = targets_dh
    targets[bg_inds, 0] = -1

    return targets, ex_max_overlaps


def scale_rois(rois, im_size, im_scale):
    im_size_scaled = np.round(im_size * im_scale)
    scale = (im_size_scaled - 1) / (im_size - 1)
    scaled_rois = rois * np.array([scale[1], scale[0], scale[1], scale[0]])
    return scaled_rois


def proposal_locate_anchors(im, anchordb):
    """ Generate anchors for different scale """
    final_anchors = []
    im_scales = []

    for target_size in cfg.TRAIN.SCALES:
        anchors, im_scale = proposal_locate_anchors_single_scale(im, target_size, anchordb)
        final_anchors.append(anchors)
        im_scales.append(im_scale)

    return final_anchors, im_scales
    

def proposal_locate_anchors_single_scale(im, target_size, anchordb):
    """ generate anchors in single scale """
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)

    im_size = im.shape
    output_size = [ anchordb['output_height_map'][im_size[0]], anchordb['output_width_map'][im_size[1]] ]

    shift_x = np.array([ i/cfg.DEDUP_BOXES for i in range(0, output_size[1]) ])
    shift_y = np.array([ i/cfg.DEDUP_BOXES for i in range(0, output_size[0]) ])
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    
    # obtain all anchor boxes
    base_anchors = anchordb['anchors']
    shift_x_y = np.array([shift_x.flatten('F'), shift_y.flatten('F'), shift_x.flatten('F'), shift_y.flatten('F')]).T
    # final_anchors = np.repeat(base_anchors, shift_x_y.shape[0], axis=0) + np.tile(shift_x_y, (base_anchors.shape[0], 1))
    final_anchors = np.tile(base_anchors, (shift_x_y.shape[0], 1)) + np.repeat(shift_x_y, base_anchors.shape[0], axis=0)

    return final_anchors, im_scale
