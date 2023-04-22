import torch
import numpy as np

from utils.box_utils import bbox_iou, xywh2xyxy
from utils.loss_utils import build_target
import torch.functional as F
def trans_vg_eval_val(pred_boxes, gt_boxes,args):
    pred_conf = pred_boxes[:, 4, :, :].contiguous().view(pred_boxes.shape[0], -1)
    max_conf, max_loc = torch.max(pred_conf, dim=1)

    pred_bbox = torch.zeros(pred_boxes.shape[0], 4).cuda()
    # pred_gi, pred_gj, pred_best_n = [], [], []

    pred_conf = pred_boxes[:, 4, :, :].data.cpu().numpy()

    max_conf_ii = max_conf.data.cpu().numpy()
    for ii in range(pred_boxes.shape[0]):
        (gj, gi) = np.where(pred_conf[ii, :, :] == max_conf_ii[ii])
        gi, gj = int(gi[0]), int(gj[0])
        # pred_gi.append(gi)
        # pred_gj.append(gj)

        pred_bbox[ii, 0] = (torch.sigmoid(pred_boxes[ii, 0, gj, gi]) + float(gi)) * (args.imsize/pred_boxes.shape[-1])
        pred_bbox[ii, 1] = (torch.sigmoid(pred_boxes[ii, 1, gj, gi]) + float(gj)) * (args.imsize/pred_boxes.shape[-1])
        pred_bbox[ii, 2] = torch.sigmoid(pred_boxes[ii, 2, gj, gi]) * args.imsize
        pred_bbox[ii, 3] = torch.sigmoid(pred_boxes[ii, 3, gj, gi]) * args.imsize


    batch_size = pred_boxes.shape[0]
    pred_boxes = torch.clamp(xywh2xyxy(pred_bbox), min=0.0)
    # gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) / float(batch_size)

    return iou,accu

def trans_vg_eval_test(pred_boxes, gt_boxes):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu_num = torch.sum(iou >= 0.5)

    return accu_num
