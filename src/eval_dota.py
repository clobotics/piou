import sys
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
import mmcv
import codecs
import pandas as pd
import glob
CENTERNET_PATH = '/datadrive/sangliang/CenterNet/src/lib/'
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts
thres=0.3
MODEL_PATH = '/datadrive/sangliang/CenterNet/exp/ctdet_angle/coco_dla_2x/model_last.pth'
TASK = 'ctdet_angle' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {} --dataset dota'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)
#ann_file='/datadrive/sangliang/CenterNet/data/retail50k/annotations/val.json'

output_folder = '/datadrive/dataset/DOTA/val/centernert_eval'
img_save_folder = '/datadrive/dataset/DOTA/val/centernert_eval/images'
mmcv.mkdir_or_exist(output_folder)
mmcv.mkdir_or_exist(img_save_folder)

img_ids = os.listdir('/datadrive/dataset/DOTA/val/images/images')
img_infos = []
predict_bboxes = []
gt_bboxes = []


CATEGORIES = [
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


# outfile = codecs.open('/datadrive/sangliang/CenterNet/data/retail50k/thres_{}_result.csv'.format(thres), 'w', encoding='utf-8')
# outfile.write('ImgUrl,xmin,ymin,width,height,angle,label,prob'+'\n')
# csv_path_list = glob.glob(os.path.join('/datadrive/sangliang/CenterNet/data/BottleTracking/task_csvs', '*.csv'))

for img_name in img_ids:

  img_path = os.path.join('/datadrive/dataset/DOTA/val/images/images',
                          img_name)
    
  img = cv2.imread(img_path)
  print(img_path) 
  ret = detector.run(img)['results']
  labels_list = ret.keys()
  print('labels_list:'+str(labels_list))
  for label in labels_list:
    print('Label:'+str(label)+':'+str(len(ret[label])))
    for bbox in ret[label]:
      label_name = CATEGORIES[label]
      outfile = codecs.open(os.path.join(output_folder, 'Task1_'+label_name+'.txt'),'a+', encoding='utf-8')
      if bbox[5] > thres:
        box = cv2.boxPoints(((bbox[0],bbox[1]),(bbox[2],bbox[3]),bbox[4]))
        box = np.int0(box)
        c=(0, 0, 255)
        cv2.drawContours(img, [box], 0, c, 2)
        box = np.reshape(box, [-1, ])
        outfile.write(img_name.split('.')[0]+' '+str('%.12f'%bbox[5])+' '+str('%.1f'%int(box[0]))+' '+str('%.1f'%int(box[1]))+' '+str('%.1f'%int(box[2]))+' '+str('%.1f'%int(box[3]))+' '+str('%.1f'%int(box[4]))+' '+str('%.1f'%int(box[5]))+' '+str('%.1f'%int(box[6]))+' '+str('%.1f'%int(box[7]))+'\n')
        
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        c=(0, 0, 255)
        txt=str(label_name)+':'+str(bbox[5])
        cv2.putText(img, txt, (int(bbox[0]), int(bbox[1]) - 2),
                  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        
        
    cv2.imwrite(os.path.join(img_save_folder,img_name ),img)
    
