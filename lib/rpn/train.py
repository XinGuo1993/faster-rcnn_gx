# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by 
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
import rpn
from rpn.config import cfg
import rpn_data_layer.roidb as rpn_roidb
from utils.timer import Timer
import numpy as np
import os
import cPickle

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, anchordb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        
        print 'Computing bounding-box regression targets...'
        self.bbox_means, self.bbox_stds = \
               rpn_roidb.add_bbox_regression_targets(roidb, anchordb)
        print 'done'

        # debug code
        # import os, cPickle, rpn
        # cache_file = os.path.join(rpn.ROOT_DIR, 'data', 'cache', 'test_nodiff_roidb.pkl')
        # with open(cache_file, 'wb') as f:
        #     cPickle.dump(roidb, f, cPickle.HIGHEST_PROTOCOL)
        # print 'write test roidb file...' 

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)
        self.solver.net.layers[0].set_anchordb(anchordb)

    def snapshot(self, final_name=None):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        anchor_size = len(cfg.ANCHOR.SCALES) * len(cfg.ANCHOR.RATIOS) 
        print anchor_size
        bbox_stds_flatten = np.tile(self.bbox_stds.reshape((self.bbox_stds.size, 1)), (anchor_size, 1))
        bbox_means_flatten = np.tile(self.bbox_means.reshape((self.bbox_means.size, 1)), (anchor_size, 1))

        net = self.solver.net
        bbox_pred_layer_name = 'proposal_bbox_pred'

        if cfg.TRAIN.BBOX_REG:
            # save original values
            orig_0 = net.params[bbox_pred_layer_name][0].data.copy()
            orig_1 = net.params[bbox_pred_layer_name][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params[bbox_pred_layer_name][0].data[...] = \
                    (net.params[bbox_pred_layer_name][0].data *
                     bbox_stds_flatten.reshape((bbox_stds_flatten.size, 1, 1, 1)))
            net.params[bbox_pred_layer_name][1].data[...] = \
                    (net.params[bbox_pred_layer_name][1].data *
                            bbox_stds_flatten.ravel() + bbox_means_flatten.ravel())

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        if not final_name:
            filename = (self.solver_param.snapshot_prefix + infix +
                        '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        else:
            filename = (self.solver_param.snapshot_prefix + infix +
                        '_' + str(final_name) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG:
            # restore net to original state
            net.params[bbox_pred_layer_name][0].data[...] = orig_0
            net.params[bbox_pred_layer_name][1].data[...] = orig_1
        
        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        train_result = {}
        timer = Timer()
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()

            # store accurate (fg/bg)
            tmp_result = self.check_error()
            train_result = self.expandTrainResult(train_result, tmp_result)

            if self.solver.iter % (100 * self.solver_param.display) == 0:
                self.show_status(self.solver.iter, train_result)
                train_result = {}
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()


    def check_error(self):
        """ Calculate the accuracy of each class """
        tmp_result = {blob_name: self.solver.net.blobs[blob_name].data for blob_name in self.solver.net.outputs}
        
        cls_score = self.solver.net.blobs['proposal_cls_score_reshape'].data
        labels = self.solver.net.blobs['labels_reshape'].data
        label_weights = self.solver.net.blobs['label_weights_reshape'].data

        accurate_fg = (cls_score[:,1:,:,:] > cls_score[:,:1,:,:]) & (labels[:,0:,:,:] == 1)
        accurate_bg = (cls_score[:,1:,:,:] <= cls_score[:,:1,:,:]) & (labels[:,0:,:,:] == 0)

        accuracy_fg = np.sum(accurate_fg * label_weights) / (np.sum(label_weights[labels[:,0:,:,:] == 1]) + cfg.EPS)
        accuracy_bg = np.sum(accurate_bg * label_weights) / (np.sum(label_weights[labels[:,0:,:,:] == 0]) + cfg.EPS)
            
        tmp_result['accuracy_fg'] = accuracy_fg
        tmp_result['accuracy_bg'] = accuracy_bg

        return tmp_result


    def expandTrainResult(self, train_result, tmp_result):
        """ Expand the dictionary which stores the train output and accuracy """
        if len(train_result) == 0:
            for blob_name in tmp_result.keys():
                train_result[blob_name] = []

        for blob_name, data in tmp_result.items():
            train_result[blob_name].append(data)

        return train_result


    def show_status(self, iter_num, train_result):
        """ Print the average result"""
        print '-' * 10 + 'Iteration: ' + str(iter_num) + '-' * 10
        print 'Training: err_fg %.3g, err_bg %.3g, loss(cls %.3g + reg %.3g)' % \
            (1 - np.mean(np.array(train_result['accuracy_fg'])), \
             1 - np.mean(np.array(train_result['accuracy_bg'])), \
             np.mean(np.array(train_result['loss_cls'])), \
             np.mean(np.array(train_result['loss_bbox']))
            )
        

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rpn_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def train_net(solver_prototxt, roidb, anchordb, output_dir,
              pretrained_model=None, max_iters=40000, final_name='final'):
    """Train a region proposal network."""
    sw = SolverWrapper(solver_prototxt, roidb, anchordb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'Save final model...'
    filename = sw.snapshot(final_name)
    print 'done'
    print 'done solving'
    
    return filename

#######################################################

def proposal_calc_output_size(imdb_name, test_net_def_file):
    """
    Calculate the output map size with different input size.  [100, MAX_SIZE]
    """
    # check cache
    cache_path = os.path.join(rpn.ROOT_DIR, 'data', 'cache')
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    # set cache file name
    cache_file_name = imdb_name
    if cfg.TRAIN.SNAPSHOT_INFIX:
        cache_file_name = cache_file_name + '_' + cfg.TRAIN.SNAPSHOT_INFIX
    cache_file_name = cache_file_name + '_output_map.pkl'

    with open(test_net_def_file, 'r') as f:
        net_name_line = f.readline().split('\"')
        if net_name_line and len(net_name_line) > 1:
            cache_file_name = net_name_line[1] + '_' + cache_file_name
        else:
            print 'Does not special the network name.'

    # check the cache file
    cache_file = os.path.join(cache_path, cache_file_name)
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            output_w_map = cPickle.load(f)
            output_h_map = cPickle.load(f)
        print '{} output map loaded from {}'.format(imdb_name, cache_file)
        return output_w_map, output_h_map

    # generate new output size map
    caffe_net = caffe.Net(test_net_def_file, caffe.TEST) 
    input_size = [ i for i in xrange(100, cfg.TRAIN.MAX_SIZE+1) ]
    output_w_map = {}
    output_h_map = {}

    for one_size in input_size:
        net_input = np.zeros((1, 3, one_size, one_size), dtype=np.float32)

        # reshape and forwarding
        caffe_net.blobs['data'].reshape(*(net_input.shape))
        caffe_net.forward(data=net_input.astype(np.float32, copy=False))
        
        # obtain the output size
        proposal_cls_score = caffe_net.blobs['proposal_cls_score'].data
        output_w_map[one_size] = np.size(proposal_cls_score, 3)
        output_h_map[one_size] = np.size(proposal_cls_score, 2)
    
    # save to cache
    with open(cache_file, 'wb') as f:
        cPickle.dump(output_w_map, f, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(output_h_map, f, cPickle.HIGHEST_PROTOCOL)
    print 'wrote anchors to {}'.format(cache_file)

    return output_w_map, output_h_map


def proposal_generate_anchors(imdb_name):
    """ """
    cache_path = os.path.join(rpn.ROOT_DIR, 'data', 'cache')
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    cache_file_name = imdb_name
    if cfg.TRAIN.SNAPSHOT_INFIX:
        cache_file_name = cache_file_name + '_' + cfg.TRAIN.SNAPSHOT_INFIX 
    cache_file_name = cache_file_name + '_anchors.pkl'
    cache_file = os.path.join(cache_path, cache_file_name)
    
    # check cache
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            anchors = cPickle.load(f)
        print '{} anchors loaded from {}'.format(imdb_name, cache_file)
    else:
        # generate new anchors
        base_anchor = [0, 0, cfg.ANCHOR.BASE_SIZE-1, cfg.ANCHOR.BASE_SIZE-1]
        ratio_anchors = anchor_ratio_jitter(base_anchor, cfg.ANCHOR.RATIOS)
        anchors = np.array([anchor_scale_jitter(anchor, cfg.ANCHOR.SCALES) for anchor in ratio_anchors])
        anchors = np.reshape(anchors, (anchors.shape[0]*anchors.shape[1], anchors.shape[2]))

        # save to cache
        with open(cache_file, 'wb') as f:
            cPickle.dump(anchors, f, cPickle.HIGHEST_PROTOCOL)
        print 'wrote anchors to {}'.format(cache_file)
    
    return anchors, cache_file


def anchor_ratio_jitter(anchor, ratios):
    """ """
    ratios = np.array(ratios, dtype=np.float32)

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + (w - 1) / 2.0
    y_ctr = anchor[1] + (h - 1) / 2.0
    size = w * h

    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    
    anchors = np.array([x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, x_ctr + (ws - 1) / 2, y_ctr + (hs -1) / 2], dtype=np.float32).T 
    # print anchors

    return anchors


def anchor_scale_jitter(anchor, scales):
    """ """
    scales = np.array(scales, dtype=np.float32)
    
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + (w - 1) / 2.0
    y_ctr = anchor[1] + (h - 1) / 2.0
    
    ws = w * scales
    hs = h * scales

    anchors = np.array([x_ctr - (ws - 1) / 2, y_ctr - (hs - 1) / 2, x_ctr + (ws - 1) / 2, y_ctr + (hs -1) / 2], dtype=np.float32).T
    # print anchors

    return anchors

