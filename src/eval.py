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
MODEL_PATH = '/datadrive/sangliang/CenterNet/exp/ctdet/coco_dla/model_best.pth'
TASK = 'ctdet' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {} --arch dla6channels_34 --dataset BottleTracking'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

ann_file='/datadrive/sangliang/CenterNet/data/BottleTracking/annotations/val.json'
#ann_file='/datadrive/sangliang/CenterNet/data/BottleTracking/tongxin_eval_dataset/annotations/eval.json'
coco = COCO(ann_file)
cat_ids = coco.getCatIds()
cat2label = {
  cat_id: i + 1
  for i, cat_id in enumerate(cat_ids)
}
img_ids = coco.getImgIds()
img_infos = []
output_folder = '/datadrive/sangliang/CenterNet/data/BottleTracking/eval'
mmcv.mkdir_or_exist(output_folder)
predict_bboxes = []
gt_bboxes = []

outfile = codecs.open('/datadrive/sangliang/CenterNet/data/BottleTracking/thres_{}_result.csv'.format(thres), 'w', encoding='utf-8')
outfile.write('ImgUrl,xmin,ymin,xmax,ymax,prob'+'\n')
csv_path_list = glob.glob(os.path.join('/datadrive/sangliang/CenterNet/data/BottleTracking/task_csvs', '*.csv'))
df = pd.DataFrame()
for csv_path in csv_path_list:
  df=df.append(pd.read_csv(csv_path, index_col=False, encoding='utf-8'))
df=df.loc[df.ImageQuality=='["qualified"]']
# gt_df = pd.read_csv('/datadrive/sangliang/CenterNet/data/BottleTracking/tongxin_eval_dataset/gt_name.csv')['image_name'].tolist()
for i in img_ids:
  info = coco.loadImgs([i])[0]
  img_path = os.path.join('/datadrive/sangliang/CenterNet/data/BottleTracking/images',
                          info['file_name'])
  # if info['file_name'] not in gt_df:
    # print(info['file_name'])
    # continue
    
  tmp_img = cv2.imread(img_path)
  left_img = tmp_img[:, :tmp_img.shape[1] // 2, :]
  right_img = tmp_img[:, tmp_img.shape[1] // 2:, :]
  img = np.concatenate((left_img, right_img), axis=2)
  ret = detector.run(img)['results']
  for bbox in ret[1]:
    if bbox[4] > thres:
      box = np.array(bbox[0:4], dtype=np.int32)
      txt = '{}{:.5f}'.format('unit', bbox[4])
      font = cv2.FONT_HERSHEY_SIMPLEX
      c=(0, 0, 255)
      cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
      cv2.rectangle(
        tmp_img, (box[0], box[1]), (box[2], box[3]), c, 2)
      outfile.write(info['flickr_url']+','+str(box[0])+','+str(box[1])+','+str(box[2])+','+str(box[3])+','+str(bbox[4])+'\n')
      cv2.rectangle(tmp_img,
                    (box[0], box[1] - cat_size[1] - 2),
                    (box[0] + cat_size[0], box[1] - 2), c, -1)
      cv2.putText(tmp_img, txt, (box[0], box[1] - 2),
                  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
  ann_ids = coco.getAnnIds(imgIds=[i])
  anns = coco.loadAnns(ids=ann_ids)
  # for k in range(len(anns)):
    # ann = anns[k]
    # box=ann['bbox']
    # cv2.rectangle(
      # tmp_img, (int(box[0]), int(box[1])), (int(box[2]+box[0]), int(box[3]+box[1])), (255,255,0), 2)
    # outfile.write(info['flickr_url'] + ',' + str(
      # box[0]) + ',' + str(box[1]) + ',' + str(box[2]+box[0]) + ',' + str(box[3]+box[1]) + ',' + str(2.0)+'\n')
  url_df = df.loc[df['ImgUrl'] == info['flickr_url']]
  for index, row in url_df.iterrows():
    ProductId = row['ProductId']
    if ProductId == 1047936:
      outfile.write(info['flickr_url'] + ',' + str(
        row['xmin']*tmp_img.shape[1]) + ',' + str(row['ymin']*tmp_img.shape[0]) + ',' + str((row['xmax'])*tmp_img.shape[1]) + ',' + str((row['ymax'])*tmp_img.shape[0]) + ',' + str(1047636.0)+'\n')
    else:
      outfile.write(info['flickr_url'] + ',' + str(
        row['xmin']*tmp_img.shape[1]) + ',' + str(row['ymin']*tmp_img.shape[0]) + ',' + str((row['xmax'])*tmp_img.shape[1]) + ',' + str((row['ymax'])*tmp_img.shape[0]) + ',' + str(2.0)+'\n')
      cv2.rectangle(
       tmp_img, (int(row['xmin']*tmp_img.shape[1]), int(row['ymin']*tmp_img.shape[0])), (int((row['xmax'])*tmp_img.shape[1]), int((row['ymax'])*tmp_img.shape[0])), (255,255,0), 2)

  cv2.imwrite(os.path.join(output_folder,info['file_name']), tmp_img)



