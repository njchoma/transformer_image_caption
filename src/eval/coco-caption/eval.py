"""
Reference: https://github.com/tylin/coco-caption
"""

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
import argparse

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

parser = argparse.ArgumentParser()
add_arg = parser.add_argument

add_arg('--gt_path', type=str, required=True,
        help='path of json file containing ground truth')
add_arg('--results_path', type=str, required=True,
            help='path of results file ')

args = parser.parse_args()
# set up file names and pathes
dataDir='.'
dataType='val2014'
algName = 'fakecap'
annFile='%s/annotations/captions_%s.json'%(dataDir,dataType)
subtypes=['results', 'evalImgs', 'eval']
[resFile, evalImgsFile, evalFile]= \
['%s/results/captions_%s_%s_%s.json'%(dataDir,dataType,algName,subtype) for subtype in subtypes]

# download Stanford models
#!./get_stanford_models.sh

coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
#cocoEval.params['image_id'] = cocoRes.getImgIds()


gt_path = args.gt_path
results_path = args.results_path

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
cocoEval.evaluate(gt_path, results_path)

# print output evaluation scores
#for metric, score in cocoEval.eval.items():
    #print '%s: %.3f'%(metric, score)
