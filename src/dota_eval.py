import sys
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
import mmcv
import codecs
import pandas as pd
import glob
import math

def rotation_bbox_to_segmentation(bbox):
    """
    :param bbox: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    cos_w = 0.5 * bbox[2] * math.cos(bbox[4])
    sin_w = 0.5 * bbox[2] * math.sin(bbox[4])
    cos_h = 0.5 * bbox[3] * math.cos(bbox[4])
    sin_h = 0.5 * bbox[3] * math.sin(bbox[4])
    x0 = bbox[0] + cos_w + sin_h
    y0 = bbox[1] - sin_w + cos_h
    x1 = bbox[0] - cos_w + sin_h
    y1 = bbox[1] + sin_w + cos_h
    x2 = bbox[0] - cos_w - sin_h
    y2 = bbox[1] + sin_w - cos_h
    x3 = bbox[0] + cos_w - sin_h
    y3 = bbox[1] - sin_w - cos_h
    corners = [x0, y0, x1, y1, x2, y2, x3, y3]
    return np.array(corners, dtype=np.float32)

CENTERNET_PATH = '/home/czm/centernet-pytorch-1.1/src/lib/'
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts
thres=0.2
MODEL_PATH = '/home/czm/centernet-pytorch-1.1/exp/ctdet_angle/dota_dla_piou_au_512/model_best.pth'
TASK = 'ctdet_angle' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {} --dataset dota'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)
#ann_file='/datadrive/sangliang/CenterNet/data/retail50k/annotations/val.json'

output_folder = '/home/czm/debug'
img_save_folder = '/home/czm/debug/images'
mmcv.mkdir_or_exist(output_folder)
mmcv.mkdir_or_exist(img_save_folder)
img_dir = '/home/czm/DOTA/test/images'
img_ids = os.listdir(img_dir)
img_infos = []
predict_bboxes = []
gt_bboxes = []


labelmap = [
  "__background",
  "baseball-diamond",
  "basketball-court",
  "bridge",
  "container-crane",
  "ground-track-field",
  "harbor",
  "helicopter",
  "large-vehicle",
  "plane",
  "roundabout",
  "ship",
  "small-vehicle",
  "soccer-ball-field",
  "storage-tank",
  "swimming-pool",
  "tennis-court"
]

result_dir = '/home/czm/centernet-pytorch-1.1/exp/ctdet_angle/dota_dla_piou_au_512/results'
if not os.path.exists(result_dir):
  os.mkdir(result_dir)
f_maps = {}
for item in labelmap:
  file_path = "{}/{}{}.txt".format(result_dir, 'Task1_', item)
  f_maps[item] = open(file_path, 'w')
print('total: ', len(img_ids))
for index, img_name in enumerate(img_ids):
  print('current...', index, img_name)
  img_path = os.path.join(img_dir,img_name)
  file_name = img_name.split('.')[0]
  img = cv2.imread(img_path)
  ret = detector.run(img)['results']
  labels_list = ret.keys()
  for label in labels_list:
    for bbox in ret[label]:
      label_name = labelmap[label]
      if bbox[5] > thres:
        corners = rotation_bbox_to_segmentation(bbox)
        info = "{} {} {} {} {} {} {} {} {} {}\n".format(
            file_name, bbox[5], corners[0], corners[1], corners[2], corners[3], corners[4], corners[5], corners[6], corners[7])
        f_maps[label_name].write(info)
for item in labelmap:
  f_maps[item].close()
        
    
