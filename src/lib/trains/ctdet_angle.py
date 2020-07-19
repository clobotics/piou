from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, PIoULoss
from models.decode import ctdet_angle_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

class CtdetAngleLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetAngleLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
      RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
      NormRegL1Loss() if opt.norm_wh else \
        RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss, angle_loss = 0, 0, 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      # print('hm', output['hm'].shape)
      # print('wh', output['wh'].shape)
      # print('angle', output['angle'].shape)
      # print('gt hm', batch['hm'].shape)
      # print('gt wh', batch['wh'].shape)
      # print('gt angle', batch['angle'].shape)
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])

      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(),
          batch['ind'].detach().cpu().numpy(),
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(),
          batch['ind'].detach().cpu().numpy(),
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)
      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4
          wh_loss += (
                         self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                      batch['dense_wh'] * batch['dense_wh_mask']) /
                         mask_weight) / opt.num_stacks
        elif opt.cat_spec_wh:
          wh_loss += self.crit_wh(
            output['wh'], batch['cat_spec_mask'],
            batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        else:
          wh_loss += self.crit_reg(
            output['wh'], batch['reg_mask'],
            batch['ind'], batch['wh']) / opt.num_stacks

      if opt.angle_weight > 0:
        angle_loss += self.crit_reg(
            output['angle'], batch['reg_mask'],
            batch['ind'], batch['angle']) / opt.num_stacks

      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                  batch['ind'], batch['reg']) / opt.num_stacks

    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss + opt.angle_weight * angle_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss,
                  'angle_loss': angle_loss}
    return loss, loss_stats

class CtdetPiouLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetPiouLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
      RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
      NormRegL1Loss() if opt.norm_wh else \
        RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.crit_piou = PIoULoss()
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, piou_loss, off_loss = 0, 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])
      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                  batch['ind'], batch['reg']) / opt.num_stacks
      # we think the center is same
      loc_t = torch.cat((batch['cxcy'], batch['wh'], batch['angle']), 2)
      loc_p = torch.cat((output['wh'], output['angle']), 1)

      piou_loss += self.crit_piou(
          loc_p, batch['reg_mask'],
          batch['ind'], loc_t) / opt.num_stacks

    loss = opt.hm_weight * hm_loss + opt.piou_weight * piou_loss + \
           opt.off_weight * off_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'piou_loss': piou_loss, 'off_loss': off_loss}
    return loss, loss_stats


class CtdetAngleTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetAngleTrainer, self).__init__(opt, model, optimizer=optimizer)

  def _get_losses(self, opt):
    if opt.piou_weight > 0:
      loss_states = ['loss', 'hm_loss', 'piou_loss', 'off_loss']
      loss = CtdetPiouLoss(opt)
    else:
      loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'angle_loss']
      loss = CtdetAngleLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    angle = output['angle']
    dets = ctdet_angle_decode(
      output['hm'], output['wh'], reg=reg, angle=angle,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det_angle'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        # center predict scores
        if dets[i, k, 5] > opt.center_thresh:
          debugger.add_rotation_bbox(dets[i, k, :5], dets[i, k, -1],
                                 dets[i, k, 5], img_id='out_pred',show_txt=False)

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 5] > opt.center_thresh:
          debugger.add_rotation_bbox(dets_gt[i, k, :5], dets_gt[i, k, -1],
                                 dets_gt[i, k, 5], img_id='out_gt',show_txt=False)

      if opt.debug == 4:
        print(batch['meta']['img_id'][i])
        a=batch['meta']['img_id'][i].detach().cpu()
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id), img_id=batch['meta']['img_id'])
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    angle = output['angle']
    dets = ctdet_angle_decode(
      output['hm'], output['wh'], reg=reg, angle=angle,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
