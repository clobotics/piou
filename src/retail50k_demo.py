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

thres = 0.2
MODEL_PATH = '/home/czm/centernet-pytorch-1.1/exp/ctdet_angle/dota_dla_512/model_best.pth'
TASK = 'ctdet_angle'  # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {} --dataset retail50k'.format(TASK,MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)
#ann_file='/datadrive/sangliang/CenterNet/data/retail50k/annotations/val.json'

output_folder = '/home/czm/debug'
img_save_folder = '/home/czm/debug/images'
mmcv.mkdir_or_exist(output_folder)
mmcv.mkdir_or_exist(img_save_folder)
img_dir = '/home/czm/Retail50K/clean_images'
img_ids = os.listdir(img_dir)
img_infos = []
predict_bboxes = []
gt_bboxes = []

CATEGORIES = [
    "__background",
    "shelf"
]


# outfile = codecs.open('/datadrive/sangliang/CenterNet/data/retail50k/thres_{}_result.csv'.format(thres), 'w', encoding='utf-8')
# outfile.write('ImgUrl,xmin,ymin,width,height,angle,label,prob'+'\n')
# csv_path_list = glob.glob(os.path.join('/datadrive/sangliang/CenterNet/data/BottleTracking/task_csvs', '*.csv'))

for index, img_name in enumerate(img_ids):
  if index > 100:
    break
  img_path = os.path.join(img_dir, img_name)

  img = cv2.imread(img_path)
  height, width, c = img.shape
  s_h = height / 512.0
  s_w = width / 512.0
  resize_img = cv2.resize(img, (512, 512))
  print(img_path)
  ret = detector.run(img)['results']
  labels_list = ret.keys()
  print('labels_list:'+str(labels_list))
  print('')
  for label in labels_list:
    print('Label:'+str(label)+':'+str(len(ret[label])))
    for bbox in ret[label]:
      label_name = CATEGORIES[label]
      if bbox[5] > thres:
        corners = rotation_bbox_to_segmentation(bbox)
        corners[0::2] = corners[0::2] * s_w
        corners[1::2] = corners[1::2] * s_h
        corners = corners.reshape(-1, 1, 2)
        corners = corners.astype(int)
        cv2.polylines(img, [corners], True, (0, 255, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        c = (0, 0, 255)
        txt = str(label_name)+':'+str(bbox[5])
        cv2.putText(img, txt, (int(bbox[0]), int(bbox[1]) - 2),
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    cv2.imwrite(os.path.join(img_save_folder, img_name), img)
