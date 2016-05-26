#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------

"""Train a Faster R-CNN network. Region Proposal Network and Fast RCNN"""

import _init_paths
from rpn import config as rpn_config
from rpn import train as rpn_train
from datasets.factory import get_imdb
import test_net as test_fast_rcnn_net
import test_rpn_net
import caffe
import argparse
import pprint
import numpy as np
import sys, os


def parse_cfg_file(file_name):
    """ load configuration for training faster-rcnn """
    import yaml
    from easydict import EasyDict as edict
    with open(file_name, 'r') as f:
        train_cfg = edict(yaml.load(f))

    return train_cfg


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--testdef', dest='test_def',
                        help='test prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=80000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='train config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='kitti_trainval', type=str)
    parser.add_argument('--mix', dest='mix_imdb_name',
                        help='dataset to mix for training',
                        default=None, type=str)
    parser.add_argument('--test', dest='test_imdb',
                        help='test the imdb at the end',
                        default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

# ================ init
    # set conf
    if args.cfg_file is not None:
        rpn_config.cfg_from_file(args.cfg_file)
    
    print('RPN using config:')
    pprint.pprint(rpn_config.cfg)

    # set up caffe
    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(rpn_config.cfg.RNG_SEED)
        caffe.set_random_seed(rpn_config.cfg.RNG_SEED)
    
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)

# ============== stage-pre
    print 'start stage-pre...'
    if not os.path.exists(args.test_def):
        raise IOError(('{:s} not found!').format(args.test_def))

    # calculate ouput size map and prepare anchors
    output_w, output_h = rpn_train.proposal_calc_output_size(args.imdb_name, 
                                                             args.test_def)
    anchors, anchors_file = rpn_train.proposal_generate_anchors(args.imdb_name)
    anchordb = {'anchors': anchors, 
                'output_width_map': output_w,
                'output_height_map': output_h
                } 
    
    print 'stage-pre done!'
    
# =============== stage-1 training rpn with imagenet parameters 
    print 'start train rpn model...'
    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)

    roidb = rpn_train.get_training_roidb(imdb)
    output_dir = rpn_config.get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    
    ### mix anothor dataset
    if args.mix_imdb_name != None:
        imdb_mix = get_imdb(args.mix_imdb_name)
        roidb_mix = rpn_train.get_training_roidb(imdb_mix)
        roidb.extend(roidb_mix)
    ### 

    rpn_model = rpn_train.train_net(args.solver, 
              roidb, anchordb, output_dir, final_name=imdb.name,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
    print 'training rpn finished!'
    
    print 're-store rpn final model...'
    # dest_path = train_cfg.stage1.model_path.format(imdb.name)
    # os.system('cp {:s} {:s}'.format(rpn_model, dest_path))
    print 'done!'
        
# =============== stage-2 testing
    final_test_imdbname = args.test_imdb
    if not final_test_imdbname:
        print 'have no image set to test!\nexit!'
        sys.exit()

    print 'start test...'

    net = caffe.Net(args.test_def, str(rpn_model), caffe.TEST)
    net.name = os.path.splitext(os.path.basename(rpn_model))[0]
    print 'Loaded network {:s}'.format(rpn_model)

    imdb = get_imdb(final_test_imdbname)
    print 'Loaded dataset `{:s}` for testing'.format(imdb.name)
    
    output_dir = get_output_dir(imdb, net)
    
    print 'start test dateset `{:s}`'.format(imdb.name)
    res_boxes = test_rpn_net.test_imdb_comp(net, imdb, anchors)
    imdb.evaluate_detections(res_boxes, output_dir)
    print 'Stage-Test: test dataset `{:s}` finished!'.format(imdb.name)
    print 'stage-Test done!'

