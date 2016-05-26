# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.car_plate
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import json
import cv2

class car_plate(datasets.imdb):
    def __init__(self, image_set, source_name, devkit_path=None):
        datasets.imdb.__init__(self, 'plate_' + source_name + '_' + image_set)
        self._source_name = source_name
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, source_name)
        self._classes = ('__background__', # always index 0
                         'plate')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_ids = self._load_image_ids()
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'Plate Data path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # print repr(index)[2:-2]
        image_path = os.path.join(self._data_path, 'Images', index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_ids(self):
        """
        Load all image ids (0-base) of this image set
        """
        image_set_file = os.path.join(self._data_path, 'ImageSets', 
                                    self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file, 'r') as f:
            image_ids = [int(x.strip()) for x in f.readlines()]
        
        return image_ids


    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        label_file = os.path.join(self._data_path, 'Annotations', 'label.dat')
        assert os.path.exists(label_file), \
                'Label file path does not exist: {}'.format(label_file)
        
        with open(label_file, 'r') as f:
            image_index = [json.loads(x.strip())['path'].encode('utf-8') 
                           for ind, x in enumerate(f.readlines()) 
                           if ind in self._image_ids]
        
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'PLATE')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        
        gt_roidb = self._load_plate_annotation(self._image_ids)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # cache_file = os.path.join(self.cache_path,
        #                           self.name + '_selective_search_roidb.pkl')

        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = cPickle.load(fid)
        #     print '{} ss roidb loaded from {}'.format(self.name, cache_file)
        #     return roidb

        gt_roidb = self.gt_roidb()
        roidb = gt_roidb
        # if self._image_set != 'test':
        #     gt_roidb = self.gt_roidb()
        #     # ss_roidb = self._load_selective_search_roidb(gt_roidb)
        #     # roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        #     roidb = gt_roidb
        # else:
        #     roidb = self._load_selective_search_roidb(None)
        # with open(cache_file, 'wb') as fid:
        #     cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        # print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # cache_file = os.path.join(self.cache_path,
        #         '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
        #         format(self.name, self.config['top_k']))

        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = cPickle.load(fid)
        #     print '{} ss roidb loaded from {}'.format(self.name, cache_file)
        #     return roidb

        # gt_roidb = self.gt_roidb()
        # ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        # roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        # with open(cache_file, 'wb') as fid:
        #     cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        # print 'wrote ss roidb to {}'.format(cache_file)

        # return roidb
        pass

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        # IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
        #                                          'selective_search_IJCV_data',
        #                                          'voc_' + self._year))
        # assert os.path.exists(IJCV_path), \
        #        'Selective search IJCV data not found at: {}'.format(IJCV_path)

        # top_k = self.config['top_k']
        # box_list = []
        # for i in xrange(self.num_images):
        #     filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
        #     raw_data = sio.loadmat(filename)
        #     box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        # return self.create_roidb_from_box_list(box_list, gt_roidb)
        pass

    def _load_plate_annotation(self, image_ids):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        label_file = os.path.join(self._data_path, 'Annotations', 'label.dat')
        
        with open(label_file, 'r') as f:
            all_keypoints = [json.loads(x.strip())['keypoints']
                         for ind, x in enumerate(f.readlines()) if ind in image_ids]
        
        gt_roidb = []

        for ind, keypoints in enumerate(all_keypoints):
            points = np.array([ [float(val) for val in keypoint.split(',')]
                                    for keypoint in keypoints ])
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            
            cls = 1
            gt_img_size = [600, 800]
            pad_size = 10
            x_scale = 1.6
            y_scale = 1.2

            # dynamic scale parameters
            if self._source_name != 'gz_10w':
                im_size = cv2.imread(self.image_path_at(ind)).shape[:2]
                x_scale = float(im_size[1]) / float(gt_img_size[1])
                y_scale = float(im_size[0]) / float(gt_img_size[0]) 

            boxes = np.array([x_min * x_scale - pad_size, y_min * y_scale - pad_size, 
                              x_max * x_scale + pad_size, y_max * y_scale + pad_size], 
                              dtype=np.float32).reshape(1, 4)
            gt_classes = np.array([cls], dtype=np.int32)
            overlaps = np.zeros((1, self.num_classes), dtype=np.float32)
            overlaps[0, cls] = 1.0
            overlaps = scipy.sparse.csr_matrix(overlaps)

            gt_roidb.append({'boxes' : boxes,
                             'gt_classes': gt_classes,
                             'gt_overlaps' : overlaps,
                             'flipped' : False
                            })
            print 'load label {:d}/{:d}'.format(ind+1, len(all_keypoints));
        return gt_roidb

    def _write_voc_results_file(self, all_boxes):
        """
        Save the final results into file, not for evaluation
        """
        label_file = os.path.join(self._data_path, 'Annotations', 'label.dat')
        assert os.path.exists(label_file), \
                'Label file path does not exist: {}'.format(label_file)
   
        with open(label_file, 'r') as f:
            all_labels = [json.loads(x.strip()) for ind, x in enumerate(f.readlines()) 
                            if ind in self._image_ids]
         
        for im_i, boxes in enumerate(all_boxes):
            all_labels[im_i]['bbox'] = (boxes[:,:-1] + 0.5).astype(int).tolist()
        
        res_output_dir = os.path.join(self._data_path, 'Results')
        if not os.path.exists(res_output_dir):
            os.makedirs(res_output_dir)

        res_file = os.path.join(res_output_dir, 'bbox_' + self._name + '.data')
        with open(res_file, 'w') as f:
            # json.dump(all_labels, f, ensure_ascii=False)
            for labels in all_labels:
                json.dump(labels, f)
                f.write('\n')
        print 'Save bounding box results to `{:s}`'.format(res_file)


    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        # self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
