#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------

"""Train a Faster R-CNN network. Region Proposal Network and Fast RCNN"""

import _init_paths
from fast_rcnn import train as fast_rcnn_train
from fast_rcnn import config as fast_rcnn_config
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
    parser.add_argument('--cfg', dest='cfg_file',
                        help='train config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--mix', dest='mix_imdb_name',
                        help='dataset to mix for training',
                        default=None, type=str)
    parser.add_argument('--stage', dest='curr_stage',
                        help='special the training stage',
                        default=0, type=int)
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
    if args.cfg_file is None:
        print 'Please the config file for training! (--cfg)'
        sys.exit()

    # load trainging config file
    train_cfg = parse_cfg_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(train_cfg)

    # set conf
    if train_cfg.common.rpn_cfg:
        rpn_config.cfg_from_file(train_cfg.common.rpn_cfg)
        print('RPN using config:')
        pprint.pprint(rpn_config.cfg)
    if train_cfg.common.fast_rcnn_cfg:
        fast_rcnn_config.cfg_from_file(train_cfg.common.fast_rcnn_cfg)
        print('Fast-RCNN using config:')
        pprint.pprint(fast_rcnn_config.cfg)

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

    if args.curr_stage < 0:
        print 'The number of stage must more than or equal to 0!'
        sys.exit()
    current_stage = args.curr_stage

# ============== stage-pre
    print 'start stage-pre...'
    # calculate ouput size map and prepare anchors
    output_w, output_h = rpn_train.proposal_calc_output_size(args.imdb_name, 
                                                    train_cfg.stage1.test_net)
    anchors, anchors_file = rpn_train.proposal_generate_anchors(args.imdb_name)
    anchordb = {'anchors': anchors, 
                'output_width_map': output_w,
                'output_height_map': output_h
                } 
    
    print 'stage-pre done!'

    
# =============== stage-1 training rpn with imagenet parameters 
    if current_stage <= 1:
        print 'start stage-1...'
        imdb = get_imdb(args.imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        
        roidb = rpn_train.get_training_roidb(imdb)
        output_dir = rpn_config.get_output_dir(imdb, None)
        print 'Output will be saved to `{:s}`'.format(output_dir)

        ### mix anothor dataset
        if args.mix_imdb_name != None:
            imdb_mix = get_imdb(args.mix_imdb_name)
            print 'Loaded dataset `{:s}` for training'.format(imdb_mix.name)
            roidb_mix = rpn_train.get_training_roidb(imdb_mix)
            roidb.extend(roidb_mix)
        ### 
 
        stage1_model = rpn_train.train_net(train_cfg.stage1.solver, 
                  roidb, anchordb, output_dir, final_name=imdb.name,
                  pretrained_model=train_cfg.common.pretrained_model,
                  max_iters=train_cfg.stage1.max_iters)
        print 'Stage-1: training rpn finished!'
    
        print 're-store stage-1 final model...'
        dest_path = train_cfg.stage1.model_path.format(imdb.name)
        os.system('cp {:s} {:s}'.format(stage1_model, dest_path))
        print 'done!'
        print 'stage-1 done!'
        
# =============== stage-2 training fast-rcnn with proposals generated by rpn
    if current_stage <= 2:
        print 'start stage-2...'
        imdb = get_imdb(args.imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        
        # Append proposl boxes
        test_rpn_net.NMS_CONFIG = train_cfg.stage1_nms
        imdb = test_rpn_net.imdb_append_proposals(train_cfg.stage1.test_net, 
               train_cfg.stage1.model_path.format(imdb.name), imdb, anchors_file)
        
        roidb = fast_rcnn_train.get_training_roidb(imdb)
        output_dir = fast_rcnn_config.get_output_dir(imdb, None)
        print 'Output will be saved to `{:s}`'.format(output_dir)

        ### mix anothor dataset
        if args.mix_imdb_name != None:
            imdb_mix = get_imdb(args.mix_imdb_name)
            print 'Loaded dataset `{:s}` for training'.format(imdb_mix.name)
            imdb_mix = test_rpn_net.imdb_append_proposals(train_cfg.stage1.test_net, 
                    train_cfg.stage1.model_path.format(imdb.name), imdb_mix, anchors_file)
            roidb_mix = fast_rcnn_train.get_training_roidb(imdb_mix)
            roidb.extend(roidb_mix)
        ### 
 
        stage2_model = fast_rcnn_train.train_net(train_cfg.stage2.solver, 
                  roidb, output_dir, final_name=imdb.name,
                  pretrained_model=train_cfg.common.pretrained_model,
                  max_iters=train_cfg.stage2.max_iters)
        print 'Stage-2: training fast-rcnn finished!'
        
        print 're-store stage-2 final model...'
        dest_path = train_cfg.stage2.model_path.format(imdb.name)
        os.system('cp {:s} {:s}'.format(stage2_model, dest_path))
        print 'done!'
        print 'stage-2 done!'
        
# =============== stage-3 training rpn (fc layers) with fast r-cnn parameters
    if current_stage <= 3:
        print 'start stage-3...'
        imdb = get_imdb(args.imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        
        roidb = rpn_train.get_training_roidb(imdb)
        output_dir = rpn_config.get_output_dir(imdb, None)
        print 'Output will be saved to `{:s}`'.format(output_dir)
        
        ### mix anothor dataset
        if args.mix_imdb_name != None:
            imdb_mix = get_imdb(args.mix_imdb_name)
            print 'Loaded dataset `{:s}` for training'.format(imdb_mix.name)
            roidb_mix = rpn_train.get_training_roidb(imdb_mix)
            roidb.extend(roidb_mix)
        ### 

        stage3_model = rpn_train.train_net(train_cfg.stage3.solver, 
                  roidb, anchordb, output_dir, final_name=imdb.name,
                  pretrained_model=train_cfg.stage2.model_path.format(imdb.name),
                  max_iters=train_cfg.stage3.max_iters)
        print 'Stage-3: fine-tune rpn finished!'
        
        print 're-store stage-3 final model...'
        dest_path = train_cfg.stage3.model_path.format(imdb.name)
        os.system('cp {:s} {:s}'.format(stage3_model, dest_path))
        print 'done!' 
        print 'stage-3 done!'

# =============== stage-4 training fast r-cnn (fc layers)
    if current_stage <= 4:
        print 'start stage-4...'
        imdb = get_imdb(args.imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    
        # Append proposl boxes
        test_rpn_net.NMS_CONFIG = train_cfg.stage3_nms
        imdb = test_rpn_net.imdb_append_proposals(train_cfg.stage3.test_net, 
               train_cfg.stage3.model_path.format(imdb.name), imdb, anchors_file)
    
        roidb = fast_rcnn_train.get_training_roidb(imdb)
        output_dir = fast_rcnn_config.get_output_dir(imdb, None)
        print 'Output will be saved to `{:s}`'.format(output_dir)

        ### mix anothor dataset
        if args.mix_imdb_name != None:
            imdb_mix = get_imdb(args.mix_imdb_name)
            print 'Loaded dataset `{:s}` for training'.format(imdb_mix.name)
            imdb_mix = test_rpn_net.imdb_append_proposals(train_cfg.stage3.test_net, 
                    train_cfg.stage3.model_path.format(imdb.name), imdb_mix, anchors_file)
            roidb_mix = fast_rcnn_train.get_training_roidb(imdb_mix)
            roidb.extend(roidb_mix)
        ### 
 
        stage4_model = fast_rcnn_train.train_net(train_cfg.stage4.solver, 
                  roidb, output_dir, final_name=imdb.name,
                  pretrained_model=train_cfg.stage2.model_path.format(imdb.name),
                  max_iters=train_cfg.stage4.max_iters)
        print 'Stage-4: fine-tune fast-rcnn finished!'
 
        print 're-store stage-4 final model...'
        dest_path = train_cfg.stage4.model_path.format(imdb.name)
        os.system('cp {:s} {:s}'.format(stage4_model, dest_path))
        print 'done!' 
        print 'stage-4 done!'
 
# =============== stage-5 testing
    if current_stage <= 5:
        print 'start stage-5...'
        final_test_imdbname = args.test_imdb
        if not final_test_imdbname:
            print 'have no image set to test!\nexit!'
            sys.exit()

        imdb = get_imdb(final_test_imdbname)
        print 'Loaded dataset `{:s}` for testing'.format(imdb.name)
        
        test_rpn_net.NMS_CONFIG = train_cfg.final_test_nms
        imdb = test_rpn_net.imdb_append_proposals(train_cfg.stage3.test_net, 
                train_cfg.stage3.model_path.format(args.imdb_name), imdb, anchors_file)
        
        print 'start test dateset `{:s}`'.format(imdb.name)
        test_fast_rcnn_net.test_imdb(train_cfg.stage4.test_net, 
                train_cfg.stage4.model_path.format(args.imdb_name), imdb)

        print 'Stage-5: test dataset `{:s}` finished!'.format(imdb.name)
        print 'stage-5 done!'

