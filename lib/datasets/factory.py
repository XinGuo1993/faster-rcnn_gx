# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.pascal_voc
import datasets.car_plate
import numpy as np

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = datasets.pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

# Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# but only returning the first k boxes
for top_k in np.arange(1000, 11000, 1000):
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
            __sets[name] = (lambda split=split, year=year, top_k=top_k:
                    _selective_search_IJCV_top_k(split, year, top_k))

# Set up plate_<source>_<split> using selective search "fast" mode
# plate_devkit_path = ''
for src_name in ['baidu', 'gz_10w']:
    for split in ['train', 'test', 'trainval', 'val', 'all', 'img1', 'img2', 'img3']:
        name = 'plate_{}_{}'.format(src_name, split)
        __sets[name] = (lambda split=split, src_name=src_name: 
                        datasets.car_plate(split, src_name))
        print __sets[name]
for src_name in ['plate']:
    for split in ['all','t1','t2','t3','t4','t5']:
        name='plate_{}_{}'.format(src_name,split)
        __sets[name]=(lambda split=split,src_name=src_name:
                      datasets.car_plate(split,src_name))
for src_name in ['plate_bad']:
    for split in ['all']:
        name='plate_{}_{}'.format(src_name,split)
        __sets[name] = (lambda split=split, src_name=src_name:
                        datasets.car_plate(split,src_name))

for src_name in ['phone','database']:
    for split in ['all','img1','img2','img4','img5']:
        name='plate_{}_{}'.format(src_name,split)
        __sets[name] = (lambda split=split, src_name=src_name:
                        datasets.car_plate(split,src_name))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    #print __sets
    #print __sets[name]
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
