# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from rpn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from rpn import test as rpn_test

def get_minibatch(roidb, anchordb, num_classes, num_anchors):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    assert(num_images == 1, \
        'proposal_generate_minibatch_fcn only support num_images == 1')

    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)

    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)
    
    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    
    # for defining the blob size
    output_size_base = [ anchordb['output_height_map'][im_blob[0].shape[1]],
                         anchordb['output_width_map'][im_blob[0].shape[2]] ]

    # Now, build the region of interest and label blobs
    labels_blob = np.zeros((0, num_anchors, output_size_base[0],\
                            output_size_base[1]), dtype=np.float32)
    label_weights_blob = np.zeros((0, num_anchors, output_size_base[0],\
                            output_size_base[1]), dtype=np.float32)
    bbox_targets_blob = np.zeros((0, 4 * num_anchors, output_size_base[0],\
                            output_size_base[1]), dtype=np.float32)
    bbox_loss_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
    all_overlaps = []

    for im_i in xrange(num_images):
        labels, label_weights, overlaps, bbox_targets, bbox_loss \
            = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                            random_scale_inds[im_i])

        # get fcn output size
        im = cv2.imread(roidb[im_i]['image'])
        im_size = np.round(np.array(im.shape[0:2]) * im_scales[im_i])
        assert(im_size[0] == im_blob[im_i].shape[1] \
            and im_size[1] == im_blob[im_i].shape[2],\
            'output size must match with the input blob size')

        output_size = [ anchordb['output_height_map'][im_size[0]], anchordb['output_width_map'][im_size[1]] ]

        # reshape the size of input blobs
        labels = np.reshape(labels, (1, output_size[1], output_size[0], num_anchors))
        label_weights = np.reshape(label_weights, (1, output_size[1], output_size[0], num_anchors))
        bbox_targets = np.reshape(bbox_targets, (1, output_size[1], output_size[0], num_anchors*4))
        bbox_loss = np.reshape(bbox_loss, (1, output_size[1], output_size[0], num_anchors*4))
        # transpose the dimension
        labels = np.transpose(labels, (0, 3, 2, 1))
        label_weights = np.transpose(label_weights, (0, 3, 2, 1))
        bbox_targets = np.transpose(bbox_targets, (0, 3, 2, 1))
        bbox_loss = np.transpose(bbox_loss, (0, 3, 2, 1))

        # Add to labels, bbox targets, and bbox loss blobs
        labels_blob = np.vstack((labels_blob, labels))
        label_weights_blob = np.vstack((label_weights_blob, label_weights))
        bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
        bbox_loss_blob = np.vstack((bbox_loss_blob, bbox_loss))
        # all_overlaps = np.hstack((all_overlaps, overlaps))
        all_overlaps.append(overlaps)
    
    # For debug visualizations
    # _vis_minibatch(im_blob, anchordb, labels_blob, label_weights_blob, all_overlaps)

    blobs = {'data': im_blob, 
             'labels': labels_blob,
             'label_weights': label_weights_blob}

    if cfg.TRAIN.BBOX_REG:
        blobs['bbox_targets'] = bbox_targets_blob
        blobs['bbox_loss_weights'] = bbox_loss_blob

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, im_scale_ind):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    overlaps = roidb['max_overlaps']
    bbox_targets = roidb['bbox_targets'][im_scale_ind]
    ex_assign_labels = bbox_targets[:, 0]
    
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(ex_assign_labels > 0)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image,
                             replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(ex_assign_labels < 0)[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image,
                             replace=False)

    # Select sampled values from various arrays:
    labels = np.zeros((bbox_targets.shape[0], 1), dtype=np.float32)
    label_weights = np.zeros((bbox_targets.shape[0], 1), dtype=np.float32)
    assert(np.all(ex_assign_labels[fg_inds] > 0))
    if fg_inds.size > 0:
        # labels[fg_inds, 0] = ex_assign_labels[fg_inds]
        # to binary label (fg/bg)
        labels[fg_inds, 0] = 1
        label_weights[fg_inds, 0] = 1
    label_weights[bg_inds, 0] = 1 
    
    # targets and target weight
    bbox_targets = bbox_targets[:, 1:]
    bbox_loss_weights = bbox_targets * 0
    if fg_inds.size > 0:
        bbox_loss_weights[fg_inds, :] = 1
    
    return labels, label_weights, overlaps, bbox_targets, bbox_loss_weights


def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_loss_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_loss_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]
    return bbox_targets, bbox_loss_weights

def _vis_minibatch(im_blob, anchordb, labels_blob, label_weights_blob, all_overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    labels_blob = labels_blob.transpose((0,3,2,1))
    labels_blob = labels_blob.reshape((labels_blob.shape[0], -1))
    label_weights_blob = label_weights_blob.transpose((0,3,2,1))
    label_weights_blob = label_weights_blob.reshape((label_weights_blob.shape[0], -1))

    print 'all overlaps len: ' + str(len(all_overlaps))
    
    for im_i in xrange(labels_blob.shape[0]):
        print 'image-{:d} overlaps len: {:d}'.format(im_i, len(all_overlaps))
        im = im_blob[im_i, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        
        im_size = im.shape
        output_size = [ anchordb['output_height_map'][im_size[0]], 
                        anchordb['output_width_map'][im_size[1]] ]
        anchors = rpn_test.proposal_locate_anchors_with_featuremap(anchordb['anchors'], 
                                                                    output_size)
        overlaps = all_overlaps[im_i][0]
        print 'scala-0 overlaps len: {:d}'.format(len(overlaps))
        if np.all(overlaps == 0):
            print '++++++ gt_rois is empty! +++++++'
            continue
        
        for rois_i in xrange(labels_blob.shape[1]):
            roi = anchors[rois_i]
            cls = labels_blob[im_i][rois_i]
            weight = label_weights_blob[im_i][rois_i]
            if weight == 0 or cls == 0:
                continue
            plt.imshow(im)
            print 'roi ind: ',rois_i , 'class: ', cls, \
                  'overlap: ', overlaps[rois_i], 'label weight: ', weight
            plt.gca().add_patch(
                plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                              roi[3] - roi[1], fill=False,
                              edgecolor='r', linewidth=3)
                )
            plt.show()

