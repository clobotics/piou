import sys
import os
import json
from xml.etree import ElementTree
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import numpy as np
import cv2
import argparse
import pandas as pd
import tqdm
import cv2
import json
import os
import urllib
import requests
import functools
import logging
import collections
import numpy as np
import sys
import time
import random
import skimage.io
START_BOUNDING_BOX_ID = 1
# PRE_DEFINE_CATEGORIES = {"unit": 1}
cats = ["shelf", ]
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
img_dir = '/DATASET/Retail50K/clean_images'
xml_dir = '/DATASET/Retail50K/xmls'
xml_list = os.listdir(xml_dir)
json_file = "/DATASET/Retail50K/annotations_retail50k_train.json"
print('total processing.... ', len(xml_list))
id = 1
bnd_id = 1
json_dict = {"images": [], "type": "instances",
             "annotations": [], "categories": []}
name_list = []
for line in open('/DATASET/Retail50K/retail50k_train.txt', 'r'):
	name_list.append(line.strip())

for name in name_list:
	filename = name + '.jpg'
	image_id = filename
	img_path = "{}/{}".format(img_dir, filename)
	img = cv2.imread(img_path)
	height, width, c = img.shape
	image = {'file_name': filename, 'height': 512, 'width': 512, 'id': id}
	json_dict['images'].append(image)
	xml_path = "{}/{}.xml".format(xml_dir, name)
	parser = etree.XMLParser(encoding='utf-8')
	xmltree = ElementTree.parse(xml_path, parser=parser).getroot()
	h_scale = 512.0 / height
	w_scale = 512.0 / width
	for object_iter in xmltree.findall('object'):
		bndbox = object_iter.find("bndbox")
		label = object_iter.find('name').text
		cat_id = cat_ids[label]
		xmin = float(bndbox.find('xmin').text) * w_scale
		ymin = float(bndbox.find('ymin').text) * h_scale
		xmax = float(bndbox.find('xmax').text) * w_scale
		ymax = float(bndbox.find('ymax').text) * h_scale
		x0 = float(bndbox.find("x0").text) * w_scale
		y0 = float(bndbox.find("y0").text) * h_scale
		x1 = float(bndbox.find("x1").text) * w_scale
		y1 = float(bndbox.find("y1").text) * h_scale
		x2 = float(bndbox.find("x2").text) * w_scale
		y2 = float(bndbox.find("y2").text) * h_scale
		x3 = float(bndbox.find("x3").text) * w_scale
		y3 = float(bndbox.find("y3").text) * h_scale
		bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
		segmentation = []
		segmentation.append(x0)
		segmentation.append(y0)
		segmentation.append(x1)
		segmentation.append(y1)
		segmentation.append(x2)
		segmentation.append(y2)
		segmentation.append(x3)
		segmentation.append(y3)
		o_width = abs(xmax - xmin)
		o_height = abs(ymax - ymin)
		ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                    id, 'bbox': [xmin, ymin, o_width, o_height],
                    'category_id': cat_id, 'id': bnd_id, 'ignore': 0,
                    'segmentation': [segmentation]}
		json_dict['annotations'].append(ann)
		bnd_id = bnd_id + 1
	id = id + 1

for cate, cid in cat_ids.items():
	cat = {'supercategory': 'none', 'id': cid, 'name': cate}
	json_dict['categories'].append(cat)
json_fp = open(json_file, 'w')
json_str = json.dumps(json_dict)
json_fp.write(json_str)
json_fp.close()
