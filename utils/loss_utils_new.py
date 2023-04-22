import torch
import numpy as np
import torch.nn.functional as F

from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from utils.misc import get_world_size
from torch.autograd import Variable
from utils.iou_loss import IOULoss
# def build_target(args, gt_bbox, pred, device):
#     batch_size = gt_bbox.size(0)
#     num_scales = len(pred)
#     coord_list, bbox_list = [], []
#     for scale_ii in range(num_scales):
#         this_stride = 32 // (2 ** scale_ii)
#         grid = args.size // this_stride
#         # Convert [x1, y1, x2, y2] to [x_c, y_c, w, h]
#         center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2
#         center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2
#         box_w = gt_bbox[:, 2] - gt_bbox[:, 0]
#         box_h = gt_bbox[:, 3] - gt_bbox[:, 1]
#         coord = torch.stack((center_x, center_y, box_w, box_h), dim=1)
#         # Normalized by the image size
#         coord = coord / args.size
#         coord = coord * grid
#         coord_list.append(coord)
#         bbox_list.append(torch.zeros(coord.size(0), 3, 5, grid, grid))
#
#     best_n_list, best_gi, best_gj = [], [], []
#     for ii in range(batch_size):
#         anch_ious = []
#         for scale_ii in range(num_scales):
#             this_stride = 32 // (2 ** scale_ii)
#             grid = args.size // this_stride
#             # gi = coord_list[scale_ii][ii,0].long()
#             # gj = coord_list[scale_ii][ii,1].long()
#             # tx = coord_list[scale_ii][ii,0] - gi.float()
#             # ty = coord_list[scale_ii][ii,1] - gj.float()
#             gw = coord_list[scale_ii][ii,2]
#             gh = coord_list[scale_ii][ii,3]
#
#             anchor_idxs = [x + 3*scale_ii for x in [0,1,2]]
#             anchors = [args.anchors_full[i] for i in anchor_idxs]
#             scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
#                 x[1] / (args.anchor_imsize/grid)) for x in anchors]
#
#             ## Get shape of gt box
#             # gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
#             # import pdb
#             # pdb.set_trace()
#
#             gt_box = torch.from_numpy(np.array([0, 0, gw.cpu().numpy(), gh.cpu().numpy()])).float().unsqueeze(0)
#             ## Get shape of anchor box
#             anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1))
#
#             ## Calculate iou between gt and anchor shapes
#             anch_ious += list(bbox_iou(gt_box, anchor_shapes))
#         ## Find the best matching anchor box
#         best_n = np.argmax(np.array(anch_ious))
#         best_scale = best_n // 3
#
#         best_grid = args.size//(32/(2**best_scale))
#         anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
#         anchors = [args.anchors_full[i] for i in anchor_idxs]
#         scaled_anchors = [ (x[0] / (args.anchor_imsize/best_grid), \
#             x[1] / (args.anchor_imsize/best_grid)) for x in anchors]
#
#         gi = coord_list[best_scale][ii,0].long()
#         gj = coord_list[best_scale][ii,1].long()
#         tx = coord_list[best_scale][ii,0] - gi.float()
#         ty = coord_list[best_scale][ii,1] - gj.float()
#         gw = coord_list[best_scale][ii,2]
#         gh = coord_list[best_scale][ii,3]
#         tw = torch.log(gw / scaled_anchors[best_n%3][0] + 1e-16)
#         th = torch.log(gh / scaled_anchors[best_n%3][1] + 1e-16)
#
#         bbox_list[best_scale][ii, best_n%3, :, gj, gi] = torch.stack([tx, ty, tw, th, torch.ones(1).to(device).squeeze()])
#         best_n_list.append(int(best_n))
#         best_gi.append(gi)
#         best_gj.append(gj)
#
#     for ii in range(len(bbox_list)):
#         bbox_list[ii] = bbox_list[ii].to(device)
#     return bbox_list, best_gi, best_gj, best_n_list


# def yolo_loss(pred_list, target, gi, gj, best_n_list, device, w_coord=5., w_neg=1./5, size_average=True):
#     mseloss = torch.nn.MSELoss(size_average=True)
#     celoss = torch.nn.CrossEntropyLoss(size_average=True)
#     num_scale = len(pred_list)
#     batch_size = pred_list[0].size(0)
#
#     pred_bbox = torch.zeros(batch_size, 4).to(device)
#     gt_bbox = torch.zeros(batch_size, 4).to(device)
#     for ii in range(batch_size):
#         pred_bbox[ii, 0:2] = torch.sigmoid(pred_list[best_n_list[ii]//3][ii, best_n_list[ii]%3,0:2, gj[ii], gi[ii]])
#         pred_bbox[ii, 2:4] = pred_list[best_n_list[ii]//3][ii, best_n_list[ii]%3, 2:4, gj[ii], gi[ii]]
#         gt_bbox[ii, :] = target[best_n_list[ii]//3][ii, best_n_list[ii]%3, :4, gj[ii], gi[ii]]
#     loss_x = mseloss(pred_bbox[:,0], gt_bbox[:,0])
#     loss_y = mseloss(pred_bbox[:,1], gt_bbox[:,1])
#     loss_w = mseloss(pred_bbox[:,2], gt_bbox[:,2])
#     loss_h = mseloss(pred_bbox[:,3], gt_bbox[:,3])
#
#     pred_conf_list, gt_conf_list = [], []
#     for scale_ii in range(num_scale):
#         pred_conf_list.append(pred_list[scale_ii][:,:,4,:,:].contiguous().view(batch_size,-1))
#         gt_conf_list.append(target[scale_ii][:,:,4,:,:].contiguous().view(batch_size,-1))
#     pred_conf = torch.cat(pred_conf_list, dim=1)
#     gt_conf = torch.cat(gt_conf_list, dim=1)
#     loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
#     return (loss_x + loss_y + loss_w + loss_h) * w_coord + loss_conf

def yolo_loss(input, target, gi, gj, w_neg=1./5, size_average=True):
    mseloss = torch.nn.MSELoss()
    celoss = torch.nn.CrossEntropyLoss()
    batch = input.size(0)

    pred_bbox = Variable(torch.zeros(batch,4).cuda())
    gt_bbox = Variable(torch.zeros(batch,4).cuda())
    for ii in range(batch):
        pred_bbox[ii, 0:2] = F.sigmoid(input[ii,0:2,gj[ii],gi[ii]])
        pred_bbox[ii, 2:4] = F.sigmoid(input[ii,2:4,gj[ii],gi[ii]])
        gt_bbox[ii, :] = target[ii,:4,gj[ii],gi[ii]]
    loss_x = mseloss(pred_bbox[:,0], gt_bbox[:,0])
    loss_y = mseloss(pred_bbox[:,1], gt_bbox[:,1])
    loss_w = mseloss(pred_bbox[:,2], gt_bbox[:,2])
    loss_h = mseloss(pred_bbox[:,3], gt_bbox[:,3])

    pred_conf_list, gt_conf_list = [], []
    pred_conf_list.append(input[:,4,:,:].contiguous().view(batch,-1))
    gt_conf_list.append(target[:,4,:,:].contiguous().view(batch,-1))
    pred_conf = torch.cat(pred_conf_list, dim=1)
    gt_conf = torch.cat(gt_conf_list, dim=1)
    # loss_conf = F.binary_cross_entropy_with_logits(pred_conf, gt_conf,reduction="sum")
    loss_conf = celoss(pred_conf,gt_conf.max(1)[1])
    return (loss_x+loss_y+loss_w+loss_h)*5 + loss_conf

def trans_vg_loss(batch_pred, batch_target,args):
    """Compute the losses related to the bounding boxes,
       including the L1 regression loss and the GIoU loss
    """
    batch_size = batch_pred[0].shape[0]
    # world_size = get_world_size()
    num_boxes = batch_size
    scales = [40]
    iou_loss = IOULoss('giou')
    loss_boxes = []
    loss_gioues = []
    for i in range(1):
        gt_param, gi, gj  = build_target(batch_target,scales[i],args)
        classfic_loss = yolo_loss(batch_pred[i], gt_param, gi, gj)
        pred_coord = torch.zeros(batch_size, 4).cuda()
        for ii in range(batch_size):
            pred_coord[ii, 0] = (F.sigmoid(batch_pred[i][ii, 0, gj[ii], gi[ii]]) + gi[ii].float()) * (args.imsize/batch_pred[i].shape[-1])
            pred_coord[ii, 1] = (F.sigmoid(batch_pred[i][ii, 1, gj[ii], gi[ii]]) + gj[ii].float()) * (args.imsize/batch_pred[i].shape[-1])
            pred_coord[ii, 2] = F.sigmoid(batch_pred[i][ii, 2, gj[ii], gi[ii]]) * args.imsize
            pred_coord[ii, 3] = F.sigmoid(batch_pred[i][ii, 3, gj[ii], gi[ii]]) * args.imsize
        pred_coord = torch.clamp(xywh2xyxy(pred_coord), min=0.0)

    # loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
    # loss_giou = 1 - torch.diag(generalized_box_iou(
    #     pred_coord,
    #     batch_target/640,
    # ))
        loss_giou = iou_loss(pred_coord,batch_target)

        # losses = {}
        # # losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        # losses['loss_bbox'] = classfic_loss
        # losses['loss_giou'] = 10*loss_giou#.sum() #/ num_boxes
        # loss.append(classfic_loss+10*loss_giou)
        loss_boxes.append(classfic_loss)
        loss_gioues.append(loss_giou*20)
    losses = {}
    losses['loss_bbox'] = sum(loss_boxes)
    losses['loss_giou'] = sum(loss_gioues)
    return losses

def build_target(raw_coord, pred, args):
    coord = Variable(torch.zeros(raw_coord.size(0), raw_coord.size(1)).cuda())
    batch  = raw_coord.size(0)
    grid = pred
    #batch, grid = raw_coord.size(0), args.size//args.gsize
    coord[:,0] = (raw_coord[:,0] + raw_coord[:,2])/(2*args.imsize)
    coord[:,1] = (raw_coord[:,1] + raw_coord[:,3])/(2*args.imsize)
    coord[:,2] = (raw_coord[:,2] - raw_coord[:,0])/(args.imsize) #width
    coord[:,3] = (raw_coord[:,3] - raw_coord[:,1])/(args.imsize) #height

    coord[:,0] = coord[:,0] * grid #center point x
    coord[:,1] = coord[:,1] * grid #center point y
    bbox=torch.zeros(coord.size(0),5,grid, grid)

    best_n_list, best_gi, best_gj = [],[],[]

    for ii in range(batch):
        #batch, grid = raw_coord.size(0), args.size//args.gsize
        gi = coord[ii,0].long() #center point x integer
        gj = coord[ii,1].long() #center point y integer
        tx = coord[ii,0] - gi.float() #center point x decimal
        ty = coord[ii,1] - gj.float() #center point y decimal
        gw = coord[ii,2] #width
        gh = coord[ii,3] #height
        try:
            bbox[ii, :, gj, gi] = torch.stack([tx, ty, gw, gh, torch.ones(1).cuda().squeeze()])
        except:
            print (raw_coord)

        #best_n_list.append(int(best_n))
        best_gi.append(gi)
        best_gj.append(gj)

    bbox = Variable(bbox.cuda())
    return bbox, best_gi, best_gj