import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from pytorch_pretrained_bert.modeling import BertModel
from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer
from utils.box_utils import xywh2xyxy
from .visual_model.darknet import *
from .dynamic_conv import DynamicBottleneck, DynamicConv2D, DynamicScale
class TransVG(nn.Module):
    def __init__(self, args):
        super(TransVG, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len
        # self.visumodel = Darknet(config_path='./models/yolov3.cfg')
        # self.visumodel.load_weights('./saved_models/yolov3.weights')
        self.convert_1024to512 = ConvBatchNormReLU(2048, hidden_dim, 1, 1, 0, 1, leaky=False)
        self.convert_512to256 = ConvBatchNormReLU(1024, hidden_dim, 1, 1, 0, 1, leaky=False)
        self.convert_256to256 = ConvBatchNormReLU(512, hidden_dim, 1, 1, 0, 1, leaky=False)

        # self.textmodel = BertModel.from_pretrained('saved_models/')
        self.visumodel = build_detr(args)
        self.textmodel = build_bert(args)

        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)
        self.text_proj_1 = nn.Linear(256, hidden_dim)
        self.text_proj_1_bias = nn.Linear(256, hidden_dim)
        self.text_proj_1_trans = nn.Linear(256, hidden_dim)
        self.text_proj_2 = nn.Linear(256, hidden_dim)
        self.text_proj_2_bias = nn.Linear(256, hidden_dim)
        self.text_proj_2_trans = nn.Linear(256, hidden_dim)
        self.text_proj_3 = nn.Linear(256, hidden_dim)
        self.text_proj_3_bias = nn.Linear(256, hidden_dim)
        self.text_proj_3_trans = nn.Linear(256, hidden_dim)

        self.v2q_layer_1 = q2v_TransformerEncoderLayer(d_model=256, nhead=8,
                                                       dim_feedforward=2048,
                                                       dropout=0.1, activation='relu',
                                                       normalize_before=False)
        self.v2q_layer_2 = q2v_TransformerEncoderLayer(d_model=256, nhead=8,
                                                       dim_feedforward=2048,
                                                       dropout=0.1, activation='relu',
                                                       normalize_before=False)
        self.v2q_layer_3 = q2v_TransformerEncoderLayer(d_model=256, nhead=8,
                                                       dim_feedforward=2048,
                                                       dropout=0.1, activation='relu',
                                                       normalize_before=False)

        self.v_pos_embed_1_trans = nn.Embedding(20 * 20, hidden_dim)
        self.v_pos_embed_2_trans = nn.Embedding(40 * 40, hidden_dim)
        self.v_pos_embed_3_trans = nn.Embedding(80 * 80, hidden_dim)
        self.l_pos_embed_1_trans = nn.Embedding(self.num_text_token, hidden_dim)
        self.l_pos_embed_2_trans = nn.Embedding(self.num_text_token, hidden_dim)
        self.l_pos_embed_3_trans = nn.Embedding(self.num_text_token, hidden_dim)

        num_total = self.num_visu_token #+ self.num_text_token
        self.vl_pos_embed = nn.Embedding(20*20+1, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        # self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.in_conv_1 = nn.Sequential(
            # nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim *2, hidden_dim*2 ),
            nn.GELU(),
            nn.Linear(hidden_dim*2 , hidden_dim),
            # nn.GELU()
        )

        self.in_conv_2 = nn.Sequential(
            # nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim *2, hidden_dim *2),
            nn.GELU(),
            nn.Linear(hidden_dim *2, hidden_dim),
            # nn.GELU()
        )

        self.in_conv_3 = nn.Sequential(
            # nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim *2, hidden_dim*2 ),
            nn.GELU(),
            nn.Linear(hidden_dim *2, hidden_dim),
            # nn.GELU()
        )


        self.fcn_out = nn.ModuleList(torch.nn.Sequential(
            ConvBatchNormReLU(256, 256 // 2, 3, 1, 1, 1, leaky=False),
            ConvBatchNormReLU(256 // 2, 256 // 4, 3, 1, 1, 1, leaky=False),
            nn.Conv2d(256 // 4, 5, kernel_size=1)) for _ in range(3))

        in_channels = 256
        bbox_subnet = []
        for _ in range(4):
            bbox_subnet_conv = DynamicBottleneck(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm="GN",
                num_groups=1,
                gate_activation="ReTanH",
                gate_activation_kargs=dict(tau=0)
            )
            bbox_subnet.append(
                DynamicScale(
                    in_channels,
                    in_channels,
                    num_convs=1,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    num_groups=1,
                    num_adjacent_scales=3,
                    resize_method="bilinear",
                    depth_module=bbox_subnet_conv,
                    gate_activation="ReTanH",
                    gate_activation_kargs=dict(tau=0)
                )
            )

        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.norms = nn.ModuleList(nn.LayerNorm(256) for _ in range(3))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.convert = nn.ModuleList(torch.nn.Sequential(
            ConvBatchNormReLU(256, 256, 3, 1, 1, 1, leaky=False),
            ConvBatchNormReLU(256, 256, 3, 1, 1, 1, leaky=False),
            nn.Conv2d(256, 256, kernel_size=1)) for _ in range(3))

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]

        # self.pos = self.poses(batch_size)
        raw_fvisu = self.visumodel(img_data.tensors)[1:]
        low_scale,_ = raw_fvisu[-1].decompose()
        mid_scale,_ = raw_fvisu[-2].decompose()
        high_scale, _ = raw_fvisu[-3].decompose()
        visu_src = [self.convert_1024to512(low_scale),
                    self.convert_512to256(mid_scale),
                    self.convert_256to256(high_scale)]


        # mask_1 = F.interpolate(img_data.mask[None].float(), size=raw_fvisu[0].shape[-2:]).to(torch.bool)[0]
        # mask_2 = F.interpolate(img_data.mask[None].float(), size=raw_fvisu[1].shape[-2:]).to(torch.bool)[0]
        # mask_3 = F.interpolate(img_data.mask[None].float(), size=raw_fvisu[2].shape[-2:]).to(torch.bool)[0]
        # mask = [mask_1,mask_2,mask_3]

        # visual backbone
        # visu_mask, visu_src = self.visumodel(img_data)
        # visu_src = self.visu_proj(visu_src)  # (N*B)xC

        # language bert
        text_fea = self.textmodel(text_data)
        text_src, text_mask = text_fea.decompose()
        assert text_mask is not None
        text_src = self.text_proj(text_src)
        # permute BxLenxC to LenxBxC
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)

        text_src_1 = self.text_proj_1(text_src).permute(1, 0, 2)
        text_src_1_bias = self.text_proj_1_bias(text_src).permute(1, 0, 2)
        text_src_1_trans = self.text_proj_1_trans(text_src).permute(1, 0, 2)

        text_src_2 = self.text_proj_2(text_src).permute(1, 0, 2)
        text_src_2_bias = self.text_proj_2_bias(text_src).permute(1, 0, 2)
        text_src_2_trans = self.text_proj_2_trans(text_src).permute(1, 0, 2)

        text_src_3 = self.text_proj_3(text_src).permute(1, 0, 2)
        text_src_3_bias = self.text_proj_3_bias(text_src).permute(1, 0, 2)
        text_src_3_trans = self.text_proj_3_trans(text_src).permute(1, 0, 2)
        # permute BxLenxC to LenxBxC

        text_src_1 = ((text_src_1 * (~text_mask).unsqueeze(-1).float()).sum(1) / (~text_mask).sum(-1).unsqueeze(
            -1).float()).unsqueeze(1).unsqueeze(1)
        text_src_2 = ((text_src_2 * (~text_mask).unsqueeze(-1).float()).sum(1) / (~text_mask).sum(-1).unsqueeze(
            -1).float()).unsqueeze(1).unsqueeze(1)
        text_src_3 = ((text_src_3 * (~text_mask).unsqueeze(-1).float()).sum(1) / (~text_mask).sum(-1).unsqueeze(
            -1).float()).unsqueeze(1).unsqueeze(1)
        text_src_1_bias = ((text_src_1_bias * (~text_mask).unsqueeze(-1).float()).sum(1) / (~text_mask).sum(-1).unsqueeze(
            -1).float()).unsqueeze(1).unsqueeze(1)
        text_src_2_bias = ((text_src_2_bias * (~text_mask).unsqueeze(-1).float()).sum(1) / (~text_mask).sum(-1).unsqueeze(
            -1).float()).unsqueeze(1).unsqueeze(1)
        text_src_3_bias = ((text_src_3_bias * (~text_mask).unsqueeze(-1).float()).sum(1) / (~text_mask).sum(-1).unsqueeze(
                -1).float()).unsqueeze(1).unsqueeze(1)

        vl_1 = self.norms[0](F.relu(torch.tanh(text_src_1) * visu_src[0].permute(0,2,3,1) + torch.tanh(text_src_1_bias)))
        vl_2 = self.norms[1](F.relu(torch.tanh(text_src_2) * visu_src[1].permute(0,2,3,1) + torch.tanh(text_src_2_bias)))
        vl_3 = self.norms[2](F.relu(torch.tanh(text_src_3) * visu_src[2].permute(0,2,3,1) + torch.tanh(text_src_3_bias)))

        # print ('adf')
        vl_trans_1 = self.v2q_layer_1(visu_src[0].flatten(2,3).permute(2,0,1), text_src_1_trans.permute(1, 0, 2), text_mask,
                                      self.v_pos_embed_1_trans.weight.unsqueeze(1).repeat(1, bs, 1),
                                      self.l_pos_embed_1_trans.weight.unsqueeze(1).repeat(1, bs, 1))
        vl_trans_2 = self.v2q_layer_2(visu_src[1].flatten(2,3).permute(2,0,1), text_src_2_trans.permute(1, 0, 2), text_mask,
                                      self.v_pos_embed_2_trans.weight.unsqueeze(1).repeat(1, bs, 1),
                                      self.l_pos_embed_1_trans.weight.unsqueeze(1).repeat(1, bs, 1))
        vl_trans_3 = self.v2q_layer_3(visu_src[2].flatten(2, 3).permute(2, 0, 1), text_src_3_trans.permute(1, 0, 2),
                                      text_mask,
                                      self.v_pos_embed_3_trans.weight.unsqueeze(1).repeat(1, bs, 1),
                                      self.l_pos_embed_1_trans.weight.unsqueeze(1).repeat(1, bs, 1))
        #
        visual_text_1 = self.in_conv_1(
            torch.cat([vl_1, vl_trans_1.permute(1, 0, 2).view(bs, 20, 20, 256)], dim=-1))#.flatten(1, 2).permute(1, 0, 2)).transpose(0, 1)

        visual_text_2 = self.in_conv_2(
            torch.cat([vl_2, vl_trans_2.permute(1, 0, 2).view(bs, 40, 40, 256)], dim=-1))#.flatten(1, 2).permute(1, 0, 2)).transpose(0, 1)

        visual_text_3 = self.in_conv_3(
            torch.cat([vl_3, vl_trans_3.permute(1, 0, 2).view(bs, 80, 80, 256)], dim=-1))#.flatten(1, 2).permute(1, 0, 2)).transpose(0, 1)

        vl = [visual_text_1.permute(0,3,1,2).contiguous(), visual_text_2.permute(0,3,1,2).contiguous(), visual_text_3.permute(0,3,1,2).contiguous()]
        a = self.bbox_subnet([vl,text_src,text_mask])

        low_scale = self.convert[0](a[0][0])
        mid_scale = self.convert[1](self.maxpool(a[0][1]))
        high_scale = self.convert[2](self.maxpool(self.maxpool(a[0][2])))

        pred_box = (low_scale + mid_scale + high_scale) / 3
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        vl_src = torch.cat([tgt_src, pred_box.flatten(2,3).permute(2,0,1)], dim=0)
        vl_mask = torch.zeros((vl_src.size(1),vl_src.size(0))).cuda().bool()
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)
        vg_hs = vg_hs[0]
        pred_box = self.bbox_embed(vg_hs).sigmoid()
        if self.training:
            return pred_box#,a[-2]
        else:
            return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class q2v_TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        tensor = tensor + pos
        return tensor

    def forward_post(self, q, k, mask, pos_q, pos_k):
        k = self.with_pos_embed(k, pos_k)
        q = self.with_pos_embed(q, pos_q)
        src2, _ = self.self_attn(q, k, k, key_padding_mask = mask)
        src = q + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src

    def forward(self, q, k, mask, pos_q, pos_k):
        return self.forward_post(q, k, mask, pos_q, pos_k)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask,pos):
        for index, layer in enumerate(self.layers):
            src = layer(src, mask, pos)
        if self.norm is not None:
            output = self.norm(src)
        return src

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        tensor = tensor + pos
        return tensor

    def forward_post(self, src, mask, pos):
        q = k = self.with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, value=src, key_padding_mask=mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, mask, pos):
        return self.forward_post(src, mask, pos)