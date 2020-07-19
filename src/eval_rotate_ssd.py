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
MODEL_PATH = '/datadrive/sangliang/CenterNet/exp/ctdet_angle/coco_dla_2x_old/model_last.pth'
TASK = 'ctdet_angle' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {} --dataset retail50k'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)
ann_file='/datadrive/sangliang/CenterNet/data/retail50k/annotations/val.json'

output_folder = '/datadrive/sangliang/CenterNet/data/retail50k/eval'
mmcv.mkdir_or_exist(output_folder)

coco = COCO(ann_file)
cat_ids = coco.getCatIds()
cat2label = {
  cat_id: i + 1
  for i, cat_id in enumerate(cat_ids)
}
img_ids = coco.getImgIds()
img_infos = []
predict_bboxes = []
gt_bboxes = []

outfile = codecs.open('/datadrive/sangliang/CenterNet/data/retail50k/thres_{}_result.csv'.format(thres), 'w', encoding='utf-8')
outfile.write('ImgUrl,prob,x0,y0,x1,y1,x2,y2,x3,y3,label'+'\n')
csv_path_list = glob.glob(os.path.join('/datadrive/sangliang/CenterNet/data/BottleTracking/task_csvs', '*.csv'))

for i in img_ids:
  info = coco.loadImgs([i])[0]
  img_path = os.path.join('/datadrive/sangliang/CenterNet/data/retail50k/clean_images',
                          info['file_name'])
    
  img = cv2.imread(img_path)

  ret = detector.run(img)['results']
  for bbox in ret[1]:
    print(bbox)
    if bbox[5] > thres:
      box = cv2.boxPoints(((bbox[0],bbox[1]),(bbox[2],bbox[3]),bbox[4]))
      box = np.int0(box)
      c=(0, 0, 255)
      cv2.drawContours(img, [box], 0, c, 2)
      box = np.reshape(box, [-1, ])
      outfile.write(info['flickr_url']+','+str('%.12f'%bbox[5])+','+str('%.1f'%int(box[0]))+','+str('%.1f'%int(box[1]))+','+str('%.1f'%int(box[2]))+','+str('%.1f'%int(box[3]))+','+str('%.1f'%int(box[4]))+','+str('%.1f'%int(box[5]))+','+str('%.1f'%int(box[6]))+','+str('%.1f'%int(box[7]))+','+str('1')+'\n')
      
      
      #outfile.write(info['flickr_url']+','+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(box[3])+','+str(bbox[4])+',1,'+str(bbox[5])+'\n')
  cv2.imwrite(os.path.join(output_folder,info['file_name'] ),img)





