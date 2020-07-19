from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from .image import transform_preds
from .ddd_utils import ddd2locrot


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan(rot[:, 6] / rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  

def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = dets.shape[2] > 16
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          transform_preds(
            dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
          .astype(np.float32)], axis=1)
    ret.append(top_preds)
  return ret

def ddd_post_process_3d(dets, calibs):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  for i in range(len(dets)):
    preds = {}
    for cls_ind in dets[i].keys():
      preds[cls_ind] = []
      for j in range(len(dets[i][cls_ind])):
        center = dets[i][cls_ind][j][:2]
        score = dets[i][cls_ind][j][2]
        alpha = dets[i][cls_ind][j][3]
        depth = dets[i][cls_ind][j][4]
        dimensions = dets[i][cls_ind][j][5:8]
        wh = dets[i][cls_ind][j][8:10]
        locations, rotation_y = ddd2locrot(
          center, alpha, dimensions, depth, calibs[0])
        bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                center[0] + wh[0] / 2, center[1] + wh[1] / 2]
        pred = [alpha] + bbox + dimensions.tolist() + \
               locations.tolist() + [rotation_y, score]
        preds[cls_ind].append(pred)
      preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
    ret.append(preds)
  return ret

def ddd_post_process(dets, c, s, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  dets = ddd_post_process_2d(dets, c, s, opt)
  dets = ddd_post_process_3d(dets, calibs)
  return dets


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

# def rotation_bboxs_to_segmentations(bboxs):
#   segmentations = np.zeros((bboxs.shape[0], 8))
#   for p in range(bboxs.shape[0]):
#     box = cv2.boxPoints(((bboxs[p][0], bboxs[p][1]), (bboxs[p][2], bboxs[p][3]), bboxs[p][4]))
#     box = np.reshape(box, [-1, ])
#     segmentations[p, 0:8] = [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]]
#   return segmentations

def rotation_bboxs_to_segmentations(bboxs):
  cos_w = 0.5 * bboxs[:, 2:3] * np.cos(bboxs[:, 4:5])
  sin_w = 0.5 * bboxs[:, 2:3] * np.sin(bboxs[:, 4:5])
  cos_h = 0.5 * bboxs[:, 3:4] * np.cos(bboxs[:, 4:5])
  sin_h = 0.5 * bboxs[:, 3:4] * np.sin(bboxs[:, 4:5])
  x0 = bboxs[:, 0:1] + cos_w + sin_h
  y0 = bboxs[:, 1:2] - sin_w + cos_h
  x1 = bboxs[:, 0:1] - cos_w + sin_h
  y1 = bboxs[:, 1:2] + sin_w + cos_h
  x2 = bboxs[:, 0:1] - cos_w - sin_h
  y2 = bboxs[:, 1:2] + sin_w - cos_h
  x3 = bboxs[:, 0:1] + cos_w - sin_h
  y3 = bboxs[:, 1:2] - sin_w - cos_h
  return np.concatenate((x0, y0, x1, y1, x2, y2, x3, y3), axis=-1)

def segmentations_to_rotation_bboxs(segmentations):
  bboxs = np.zeros((segmentations.shape[0], 5))
  for p in range(segmentations.shape[0]):
    box = np.int0(segmentations[p])
    box = box.reshape([-1, 2])
    rect1 = cv2.minAreaRect(box)
    rwidth, rheight = rect1[1]
    rangle = rect1[2]
    if rwidth > rheight:
      rangle = np.abs(rangle)
    else:
      temp = rwidth
      rwidth = rheight
      rheight = temp
      rangle = np.abs(rangle) + 90
    bboxs[p, :] = [rect1[0][0], rect1[0][1], rwidth, rheight, rangle / 180.0 * 3.141593]

  return bboxs
# def segmentations_to_rotation_bboxs(segmentations):
#   bboxs = np.zeros((segmentations.shape[0], 5))
#   for p in range(segmentations.shape[0]):
#     box = np.int0(segmentations[p])
#     box = box.reshape([-1, 2])
#     rect1 = cv2.minAreaRect(box)
#     bboxs[p, :] = [rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]]

#   return bboxs

def ctdet_angle_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    # Since there is angle involved, this method is failed
    # dets[i, :, :2] = transform_preds(
    #       dets[i, :, 0:2], c[i], s[i], (w, h))
    # dets[i, :, 2:4] = transform_preds(
    #       dets[i, :, 2:4], c[i], s[i], (w, h))
    segmentations = rotation_bboxs_to_segmentations(dets[i, :, :5])
    segmentations[ :, :2] = transform_preds(
          segmentations[ :, 0:2], c[i], s[i], (w, h))
    segmentations[ :, 2:4] = transform_preds(
          segmentations[:, 2:4], c[i], s[i], (w, h))
    segmentations[ :, 4:6] = transform_preds(
          segmentations[ :, 4:6], c[i], s[i], (w, h))
    segmentations[ :, 6:8] = transform_preds(
          segmentations[ :, 6:8], c[i], s[i], (w, h))
    bboxs = segmentations_to_rotation_bboxs(segmentations)
    dets[i, :, :5] = bboxs[:, :5]

    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :5].astype(np.float32),
        dets[i, inds, 5:6].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret


def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5], 
       pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret
