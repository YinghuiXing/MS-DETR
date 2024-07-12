import os
import torch
import math
import copy

import torch.nn.functional as F
from .segmentation import MHAttentionMap, MaskHeadSmallConv

from torch import nn
from util.misc import (NestedTensor, inverse_sigmoid)
from .racn import Group


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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


class MS_DETR(nn.Module):
    """ This is the Deformable-DETR for multi_spectral pedestrian detection """

    def get_input_proj(self, strides, num_channels):
        if num_channels is None:
            return None

        input_proj_list = []
        if self.num_feature_levels > 1:
            backbone_out_levels = len(strides)
            for lvl in range(backbone_out_levels):
                in_channels = num_channels[lvl]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for lvl in range(self.num_feature_levels - backbone_out_levels):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
        else:
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(num_channels[0], self.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, self.hidden_dim),
            ))
        input_proj = nn.ModuleList(input_proj_list)

        for proj in input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        return nn.ModuleList(input_proj_list)

    def _init_object_queries(self):
        if self.use_dab:
            self.query_embed = nn.Embedding(self.num_queries, 4)
            if self.random_xy:
                self.query_embed.weight.data[:, :2].uniform_(0, 1)
                self.query_embed.weight.data[:, :2] = inverse_sigmoid(self.query_embed.weight.data[:, :2])
                self.query_embed.weight.data[:, :2].requires_grad = False
        else:
            self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim * 2)

            if not self.share_object_queries:
                self.query_embed_rgb = nn.Embedding(self.num_queries, self.hidden_dim * 2)
                self.query_embed_t = nn.Embedding(self.num_queries, self.hidden_dim * 2)

        self.tgt_embed = nn.Embedding(self.num_queries, self.hidden_dim) if self.content_embedding and not self.modality_crossover else None
        self.tgt_embed_rgb = nn.Embedding(self.num_queries, self.hidden_dim) if (self.rgb_branch and self.content_embedding) or (self.modality_crossover) else None
        self.tgt_embed_t = nn.Embedding(self.num_queries, self.hidden_dim) if (self.t_branch and self.content_embedding) or (self.modality_crossover) else None

    def _init_det_head(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        if not self.modality_crossover:
            self.class_embed, self.bbox_embed = self._get_det_head(bias_value)

        if self.rgb_branch or self.modality_crossover:
            self.class_embed_rgb, self.bbox_embed_rgb = self._get_det_head(bias_value)

        if self.t_branch or self.modality_crossover:
            self.class_embed_t, self.bbox_embed_t = self._get_det_head(bias_value)
        
        if self.distill:
            self.class_embed_distill, self.bbox_embed_distill = self._get_det_head(bias_value)

        if self.with_box_refine:
            self.transformer.decoder.bbox_embed = self.bbox_embed if not self.modality_crossover else None
            self.transformer.decoder.bbox_embed_rgb = self.bbox_embed_rgb if (self.rgb_branch and not self.share_object_queries) or self.modality_crossover else None
            self.transformer.decoder.bbox_embed_t = self.bbox_embed_t if (self.t_branch and not self.share_object_queries) or self.modality_crossover else None

            if self.distill:
                self.transformer.decoder_distill.bbox_embed = self.bbox_embed_distill

            self.transformer.decoder.class_embed_rgb = self.class_embed_rgb if self.modality_crossover else None
            self.transformer.decoder.class_embed_t = self.class_embed_t if self.modality_crossover else None

        if self.two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def _init_seg_head(self, num_channels_rgb, num_channels_t, num_channels_fusion, dropout):
        if self.segmentation:
            if self.segmentation_stage in ('backbone', 'encoder'):
                seg_head_list_rgb = []
                seg_head_list_t = []
                seg_head_list_fusion = []

                for i in range(3):
                    if num_channels_rgb is not None:
                        seg_head_list_rgb.append(nn.Sequential(
                            nn.Conv2d(num_channels_rgb[i] if self.segmentation_stage == 'backbone' else self.hidden_dim, 64, kernel_size=3, padding=1, bias=False),
                            torch.nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                            torch.nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, self.num_classes - 1, kernel_size=1, bias=False)))

                    if num_channels_t is not None:
                        seg_head_list_t.append(nn.Sequential(
                            nn.Conv2d(num_channels_t[i] if self.segmentation_stage == 'backbone' else self.hidden_dim, 64, kernel_size=3, padding=1, bias=False),
                            torch.nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                            torch.nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, self.num_classes - 1, kernel_size=1, bias=False)))

                    if num_channels_fusion is not None:
                        seg_head_list_fusion.append(nn.Sequential(
                            nn.Conv2d(num_channels_fusion[i] if self.segmentation_stage == 'backbone' else self.hidden_dim, 64, kernel_size=3, padding=1, bias=False),
                            torch.nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                            torch.nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, self.num_classes - 1, kernel_size=1, bias=False)))

                self.seg_head_rgb = nn.ModuleList(seg_head_list_rgb) if len(seg_head_list_rgb) else None
                self.seg_head_t = nn.ModuleList(seg_head_list_t) if len(seg_head_list_t) else None
                self.seg_head_fusion = nn.ModuleList(seg_head_list_fusion) if len(seg_head_list_fusion) else None
            else:
                self.bbox_attention_rgb = MHAttentionMap(self.transformer.d_model, self.transformer.d_model,
                                                         self.transformer.n_heads, dropout=dropout)
                self.bbox_attention_t = copy.deepcopy(self.bbox_attention_rgb)
                self.mask_head_rgb = MaskHeadSmallConv(self.transformer.d_model + self.transformer.n_heads,
                                                       [1024, 512, 256], self.transformer.d_model)
                self.mask_head_t = copy.deepcopy(self.mask_head_rgb)

    def _get_det_head(self, bias_value):
        class_embed = nn.Linear(self.hidden_dim if not self.split_cls_reg else self.hidden_dim // 2, self.num_classes)
        bbox_embed = MLP(self.hidden_dim if not self.split_cls_reg else self.hidden_dim // 2, self.hidden_dim if not self.split_cls_reg else self.hidden_dim // 2, 4, 3)

        class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)

        if self.with_box_refine:
            class_embed = _get_clones(class_embed, self.num_pred)
            bbox_embed = _get_clones(bbox_embed, self.num_pred)
            nn.init.constant_(bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        else:
            nn.init.constant_(bbox_embed.layers[-1].bias.data[2:], -2.0)
            class_embed = nn.ModuleList([class_embed for _ in range(self.num_pred)])
            bbox_embed = nn.ModuleList([bbox_embed for _ in range(self.num_pred)])

        return class_embed, bbox_embed

    def _init_rec_module(self):
        if self.distill and (self.rec_fusion or self.rec_another):
            rec_module = nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim * 2, 1),
                Group(num_channels=self.hidden_dim * 2, num_blocks=20, reduction=16, use_kernal_3=self.use_kernal_3),
                nn.Conv1d(self.hidden_dim * 2, self.hidden_dim, 1)
            )

            if self.with_box_refine:
                self.rec_module = _get_clones(rec_module, self.num_pred)
                self.transformer.decoder_distill.rec_module = self.rec_module            
            else:
                self.rec_module = rec_module
            
    def _init_requires_grad(self):
        if self.distill:
            for name, parameter in self.named_parameters():
                if 'distill' in name or 'rec_module' in name:
                    parameter.requires_grad_(True)
                else:
                    parameter.requires_grad_(False)
        
        if self.rec_flag and self.rec_freeze:
            for name, parameter in self.named_parameters():
                if 'rec_module' in name:
                    parameter.requires_grad_(True)
                else:
                    parameter.requires_grad_(False)

    def __init__(self, backbone, transformer, cfg, num_classes, aux_loss, distill, distill_modality_rgb, distill_fusion_features, rec_fusion, rec_another, rec_use_kernal_3, rec_flag, rec_freeze):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            cfg: config
        """
        super().__init__()

        self.num_queries = cfg.num_queries
        self.rgb_branch = cfg.rgb_branch
        self.t_branch = cfg.t_branch
        self.modality_crossover = cfg.modality_crossover
        self.share_object_queries = cfg.share_object_queries  # 当三个分支共享oq时，object queries一致，并且object queries的更新也一致
        self.num_feature_levels = cfg.num_feature_levels
        self.use_dab = cfg.use_dab
        self.content_embedding = cfg.content_embedding
        self.random_xy = cfg.random_xy
        self.with_box_refine = cfg.with_box_refine
        self.split_cls_reg = cfg.split_cls_reg
        self.segmentation = cfg.SEGMENTATION.flag
        self.freeze_detr = cfg.SEGMENTATION.freeze_detr
        self.segmentation_stage = cfg.SEGMENTATION.stage
        self.segmentation_re_weight = cfg.SEGMENTATION.re_weight
        self.segmentation_re_weight_detach = cfg.SEGMENTATION.re_weight_detach
        self.distill = distill
        self.distill_modality_rgb = distill_modality_rgb
        self.distill_fusion_features = distill_fusion_features
        self.rec_fusion = rec_fusion
        self.rec_another = rec_another
        self.use_kernal_3 = rec_use_kernal_3
        self.rec_flag = rec_flag
        self.rec_freeze = rec_freeze
        self.analysis_weights = cfg.analysis_weights

        self.num_classes = num_classes
        self.aux_loss = aux_loss

        self.transformer = transformer
        self.backbone = backbone

        self.hidden_dim = transformer.d_model
        self.two_stage = cfg.two_stage
        self.num_pred = transformer.decoder.num_layers + 1 if self.two_stage else transformer.decoder.num_layers

        #  构造映射层
        self.input_proj_rgb = self.get_input_proj(backbone.strides, backbone.num_channels_rgb)
        self.input_proj_t = self.get_input_proj(backbone.strides, backbone.num_channels_t) if not backbone.backbone_share else self.input_proj_rgb
        self.input_proj_fusion = self.get_input_proj(backbone.strides, backbone.num_channels_fusion)

        # 构造object queries
        if not self.two_stage:
            self._init_object_queries()

        #  构造分类头和回归头
        self._init_det_head()
        self._init_rec_module()

        self._init_seg_head(backbone.num_channels_rgb, backbone.num_channels_t, backbone.num_channels_fusion, cfg.SEGMENTATION.dropout)

        self._init_requires_grad()

    def _split_features(self, features, pos, modality, img_mask):
        if not len(features):
            return [], [], []
        srcs, masks = list(), list()

        if modality == 'fusion':
            input_proj = self.input_proj_fusion
        elif modality == 'rgb':
            input_proj = self.input_proj_rgb
        elif modality == 't':
            input_proj = self.input_proj_t
        else:
            raise RuntimeError

        for lvl in range(len(features)):
            src, mask = features[lvl].decompose()
            srcs.append(input_proj[lvl](src))
            masks.append(mask)

        if self.num_feature_levels > len(features):
            _len_srcs = len(srcs)
            for lvl in range(_len_srcs, self.num_feature_levels):
                if lvl == _len_srcs:
                    src = input_proj[lvl](features[-1].tensors)
                else:
                    src = input_proj[lvl](srcs[-1])

                mask = F.interpolate(img_mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_lvl = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)

                srcs.append(src)
                masks.append(mask)
                pos.append(pos_lvl)

        return srcs, masks, pos

    def forward(self, samples_rgb: NestedTensor, samples_t: NestedTensor, targets=None):
        """ The forward expects two NestedTensors, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        features_rgb, features_t, features_fusion, pos_rgb, pos_t, pos_fusion = self.backbone(samples_rgb, samples_t, targets)
        
        pred_masks_rgb = list()
        pred_masks_t = list()
        pred_masks_fusion = list()

        if self.segmentation and self.segmentation_stage == 'backbone':
            if len(features_rgb):
                for i in range(len(features_rgb)):
                    curr_mask_rgb = self.seg_head_rgb[i](features_rgb[i].tensors)
                    if self.segmentation_re_weight:
                        features_rgb[i].tensors = features_rgb[i].tensors * curr_mask_rgb.sigmoid() \
                            if not self.segmentation_re_weight_detach \
                            else features_rgb[i].tensors * curr_mask_rgb.sigmoid().detach()
                    pred_masks_rgb.append(curr_mask_rgb)

            if len(features_t):
                for i in range(len(features_t)):
                    curr_mask_t = self.seg_head_t[i](features_t[i].tensors)
                    if self.segmentation_re_weight:
                        features_t[i].tensors = features_t[i].tensors * curr_mask_t.sigmoid() \
                            if not self.segmentation_re_weight_detach \
                            else features_t[i].tensors * curr_mask_t.sigmoid().detach()
                    pred_masks_t.append(curr_mask_t)

            if len(features_fusion):
                for i in range(len(features_fusion)):
                    curr_mask_fusion = self.seg_head_fusion[i](features_fusion[i].tensors)
                    if self.segmentation_re_weight:
                        features_fusion[i].tensors = features_fusion[i].tensors * curr_mask_fusion.sogmoid() \
                            if not self.segmentation_re_weight_detach \
                            else features_fusion[i].tensors * curr_mask_fusion.sogmoid().detach()
                    pred_masks_fusion.append(curr_mask_fusion)

        srcs_rgb, masks_rgb, pos_rgb = self._split_features(features_rgb, pos_rgb, 'rgb', samples_rgb.mask if samples_rgb is not None else None)
        srcs_t, masks_t, pos_t = self._split_features(features_t, pos_t, 't', samples_t.mask if samples_t is not None else None)
        srcs_fusion, masks_fusion, pos_fusion = self._split_features(features_fusion, pos_fusion, 'fusion', samples_rgb.mask if samples_rgb is not None else samples_t.mask)

        transformer_input = dict()
        transformer_input['srcs_rgb'] = srcs_rgb
        transformer_input['masks_rgb'] = masks_rgb
        transformer_input['pos_rgb'] = pos_rgb

        transformer_input['srcs_t'] = srcs_t
        transformer_input['masks_t'] = masks_t
        transformer_input['pos_t'] = pos_t

        transformer_input['srcs_fusion'] = srcs_fusion
        transformer_input['masks_fusion'] = masks_fusion
        transformer_input['pos_fusion'] = pos_fusion

        transformer_input['query_embed'] = self.query_embed.weight if not self.two_stage else None
        transformer_input['query_embed_rgb'] = self.query_embed_rgb.weight if not self.two_stage and not self.share_object_queries else None
        transformer_input['query_embed_t'] = self.query_embed_t.weight if not self.two_stage and not self.share_object_queries else None

        if self.modality_crossover or self.two_stage:
            transformer_input['tgt_embed'] = None
        else:
            transformer_input['tgt_embed'] = self.tgt_embed.weight if self.tgt_embed is not None else torch.zeros((self.num_queries, self.hidden_dim), device=transformer_input['query_embed'].device)
        if not self.two_stage:
            transformer_input['tgt_embed_rgb'] = self.tgt_embed_rgb.weight if self.tgt_embed_rgb is not None else (torch.zeros((self.num_queries, self.hidden_dim), device=transformer_input['query_embed'].device) if self.rgb_branch else None)
            transformer_input['tgt_embed_t'] = self.tgt_embed_t.weight if self.tgt_embed_t is not None else (torch.zeros((self.num_queries, self.hidden_dim), device=transformer_input['query_embed'].device) if self.t_branch else None)

        hs, hs_rgb, hs_t, memory_rec, memory_teacher, \
        init_reference, init_reference_rgb, init_reference_t, inter_references_distill_gt, \
        inter_references, inter_references_rgb, inter_references_t, \
        points_list, weights_list, \
        memory, spatial_shapes, enc_outputs_class, enc_outputs_coord_unact, \
        hs_distill, hs_rec_another, hs_distill_before_rec, init_reference_distill, inter_references_distill, inter_references_distill_pred, points_list_distill, weights_list_distill = self.transformer(transformer_input)

        outputs_classes = []
        outputs_coords = []

        outputs_classes_rgb = []
        outputs_coords_rgb = []

        outputs_classes_t = []
        outputs_coords_t = []

        outputs_classes_distill = []
        outputs_coords_distill = []

        if hs is not None:
            pred_nums = hs.shape[0]

        if hs_rgb is not None:
            pred_nums = hs_rgb.shape[0]

        if hs_t is not None:
            pred_nums = hs_t.shape[0]

        if hs_distill is not None:
            pred_nums = hs_distill.shape[0]

        for lvl in range(pred_nums):
            if lvl == 0:
                reference = init_reference if hs is not None or self.modality_crossover else None
                reference_rgb = init_reference_rgb if init_reference_rgb is not None else init_reference if hs_rgb is not None and not self.modality_crossover else None
                reference_t = init_reference_t if init_reference_t is not None else init_reference if hs_t is not None and not self.modality_crossover else None
                reference_distill = init_reference_distill if self.distill else None
            else:
                reference = inter_references[lvl - 1] if inter_references is not None else None

                if inter_references_rgb is not None:
                    reference_rgb = inter_references_rgb[lvl - 1]
                elif hs_rgb is not None:
                    reference_rgb = inter_references[lvl - 1]
                else:
                    reference_rgb = None

                if inter_references_t is not None:
                    reference_t = inter_references_t[lvl - 1]
                elif hs_t is not None:
                    reference_t = inter_references[lvl - 1]
                else:
                    reference_t = None

                if self.distill:
                    reference_distill = inter_references_distill[lvl - 1]
                else:
                    reference_distill = None

            reference = inverse_sigmoid(reference) if reference is not None else None
            reference_rgb = inverse_sigmoid(reference_rgb) if reference_rgb is not None else None
            reference_t = inverse_sigmoid(reference_t) if reference_t is not None else None
            reference_distill = inverse_sigmoid(reference_distill) if reference_distill is not None else None

            if reference is not None and not self.modality_crossover:
                if not self.split_cls_reg:
                    outputs_class = self.class_embed[lvl](hs[lvl])
                    tmp = self.bbox_embed[lvl](hs[lvl])
                else:
                    outputs_class = self.class_embed[lvl](hs[lvl][..., self.hidden_dim // 2 :])  # 分类特征置于后
                    tmp = self.bbox_embed[lvl](hs[lvl][..., :self.hidden_dim // 2])  # 回归特征置于前
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

            if reference_distill is not None:
                if not self.split_cls_reg:
                    outputs_class_distill = self.class_embed_distill[lvl](hs_distill[lvl])
                    tmp_distill = self.bbox_embed_distill[lvl](hs_distill[lvl])
                else:
                    outputs_class_distill = self.class_embed_distill[lvl](hs_distill[lvl][..., self.hidden_dim // 2 :])  # 分类特征置于后
                    tmp_distill = self.bbox_embed_distill[lvl](hs_distill[lvl][..., :self.hidden_dim // 2])  # 回归特征置于前
                if reference_distill.shape[-1] == 4:
                    tmp_distill += reference_distill
                else:
                    tmp_distill[..., :2] += reference_distill
                outputs_coord_distill = tmp_distill.sigmoid()
                outputs_classes_distill.append(outputs_class_distill)
                outputs_coords_distill.append(outputs_coord_distill)

            if self.modality_crossover:
                outputs_class_rgb = self.class_embed_rgb[lvl](hs_rgb[lvl])
                tmp_rgb = self.bbox_embed_rgb[lvl](hs_rgb[lvl])

                outputs_class_t = self.class_embed_t[lvl](hs_t[lvl])
                tmp_t = self.bbox_embed_t[lvl](hs_t[lvl])

                pede_logit = torch.stack([outputs_class_rgb[..., 0], outputs_class_t[..., 0]], dim=-1)
                tmp_weight = F.softmax(pede_logit, -1)
                tmp_weight_rgb = tmp_weight[..., 0]
                tmp_weight_rgb = tmp_weight_rgb[..., None]
                tmp_weight_t = tmp_weight[..., 1]
                tmp_weight_t = tmp_weight_t[..., None]
                tmp = tmp_rgb * tmp_weight_rgb + tmp_t * tmp_weight_t
                outputs_class = outputs_class_rgb * tmp_weight_rgb + outputs_class_t * tmp_weight_t

                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)
            else:
                if self.rgb_branch and reference_rgb is not None:
                    if not self.split_cls_reg:
                        outputs_class_rgb = self.class_embed_rgb[lvl](hs_rgb[lvl])
                        tmp_rgb = self.bbox_embed_rgb[lvl](hs_rgb[lvl])
                    else:
                        outputs_class_rgb = self.class_embed_rgb[lvl](hs_rgb[lvl][..., self.hidden_dim // 2 :])  # 分类特征置于后
                        tmp_rgb = self.bbox_embed_rgb[lvl](hs_rgb[lvl][..., :self.hidden_dim // 2])  # 回归特征置于前

                    if reference_rgb.shape[-1] == 4:
                        tmp_rgb += reference_rgb
                    else:
                        tmp_rgb[..., :2] += reference_rgb
                    outputs_coord_rgb = tmp_rgb.sigmoid()

                    outputs_classes_rgb.append(outputs_class_rgb)
                    outputs_coords_rgb.append(outputs_coord_rgb)

                if self.t_branch and reference_t is not None:
                    if not self.split_cls_reg:
                        outputs_class_t = self.class_embed_t[lvl](hs_t[lvl])
                        tmp_t = self.bbox_embed_t[lvl](hs_t[lvl])
                    else:
                        outputs_class_t = self.class_embed_t[lvl](hs_t[lvl][..., self.hidden_dim // 2 :])
                        tmp_t = self.bbox_embed_t[lvl](hs_t[lvl][..., :self.hidden_dim // 2])

                    if reference_t.shape[-1] == 4:
                        tmp_t += reference_t
                    else:
                        tmp_t[..., :2] += reference_t
                    outputs_coord_t = tmp_t.sigmoid()

                    outputs_classes_t.append(outputs_class_t)
                    outputs_coords_t.append(outputs_coord_t)
                
        outputs_class = torch.stack(outputs_classes) if len(outputs_classes) else None
        outputs_coord = torch.stack(outputs_coords) if len(outputs_coords) else None

        outputs_class_rgb = torch.stack(outputs_classes_rgb) if len(outputs_classes_rgb) else None
        outputs_coord_rgb = torch.stack(outputs_coords_rgb) if len(outputs_coords_rgb) else None

        outputs_class_t = torch.stack(outputs_classes_t) if len(outputs_classes_t) else None
        outputs_coord_t = torch.stack(outputs_coords_t) if len(outputs_coords_t) else None

        outputs_class_distill = torch.stack(outputs_classes_distill) if len(outputs_classes_distill) else None
        outputs_coord_distill = torch.stack(outputs_coords_distill) if len(outputs_coords_distill) else None

        out = None
        out_rgb = None
        out_t = None
        out_distill = None

        if outputs_class is not None:
            out = dict()
            out['pred_logits'] = outputs_class[-1]
            out['pred_boxes'] = outputs_coord[-1]
            out['points_list'] = points_list
            out['weights_list'] = weights_list

            if self.analysis_weights:
                c_bs, c_nq, c_h, c_l, c_n = weights_list[0].shape
                weights_rgb_total = torch.zeros((c_bs, c_nq), device=weights_list[0].device, dtype=weights_list[0].dtype)
                weights_t_total = torch.zeros((c_bs, c_nq), device=weights_list[0].device, dtype=weights_list[0].dtype)

                if weights_list is not None:
                    for i, weights in enumerate(weights_list):
                        weights_4_rgb = weights[:, :, :, :4, :] 
                        weights_4_t = weights[:, :, :, 4:, :]

                        weights_rgb_total += torch.sum(weights_4_rgb, dim=(2, 3, 4)) / 8
                        weights_t_total += torch.sum(weights_4_t, dim=(2, 3, 4)) / 8

                    out['weights_rgb'] = weights_rgb_total / len(weights_list)
                    out['weights_t'] = weights_t_total / len(weights_list)

            out['init_reference'] = init_reference
            out['inter_references'] = inter_references
            out['hs'] = hs
            out['memory_rec'] = memory_rec if memory_rec is not None else None
            out['memory_teacher'] = memory_teacher.detach() if memory_teacher is not None else None

        if outputs_class_rgb is not None:
            out_rgb = {'pred_logits': outputs_class_rgb[-1], 'pred_boxes': outputs_coord_rgb[-1]}
            out_rgb['hs_rgb'] = hs_rgb

        if outputs_class_t is not None:
            out_t = {'pred_logits': outputs_class_t[-1], 'pred_boxes': outputs_coord_t[-1]}
            out_t['hs_t'] = hs_t
        
        if outputs_class_distill is not None:
            out_distill = dict()
            out_distill['pred_logits'] = outputs_class_distill[-1]
            out_distill['pred_boxes'] = outputs_coord_distill[-1]
            out_distill['inter_reference_teacher'] = outputs_coord[-1]
            out_distill['inter_reference_student'] = outputs_coord_distill[-1]
            out_distill['points_list'] = points_list_distill
            out_distill['weights_list'] = weights_list_distill
            out_distill['init_reference'] = init_reference_distill
            out_distill['inter_references'] = inter_references_distill
            out_distill['hs'] = hs_distill

            if self.rec_fusion:
                out_distill['hs_student'] = hs_distill[-1]
            elif self.rec_another:
                out_distill['hs_student'] = hs_rec_another[-1]

            if self.rec_fusion:
                out_distill['hs_teacher'] = hs[-1]
            elif self.rec_another:
                out_distill['hs_teacher'] = hs_t[-1] if self.distill_modality_rgb else hs_rgb[-1]
            else:
                out_distill['hs_teacher'] = hs[-1] if self.distill_fusion_features else (hs_rgb[-1] if self.distill_modality_rgb else hs_t[-1])

        if self.aux_loss:
            if outputs_class is not None:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            if outputs_class_rgb is not None:
                out_rgb['aux_outputs'] = self._set_aux_loss(outputs_class_rgb, outputs_coord_rgb)
            if outputs_class_t is not None:
                out_t['aux_outputs'] = self._set_aux_loss(outputs_class_t, outputs_coord_t)
            if outputs_class_distill is not None:
                if self.rec_fusion:
                    cur_hs_student = hs_distill
                elif self.rec_another:
                    cur_hs_student = hs_rec_another
                else:
                    cur_hs_student = hs_distill

                if self.rec_fusion:
                    cur_hs_teacher = hs
                elif self.rec_another:
                    cur_hs_teacher = hs_t if self.distill_modality_rgb else hs_rgb
                else:
                    cur_hs_teacher = hs if self.distill_fusion_features else (hs_rgb if self.distill_modality_rgb else hs_t)

                out_distill['aux_outputs'] = self._set_aux_loss_4_distill(outputs_class_distill, outputs_coord_distill, 
                                                                          inter_references_distill_gt, inter_references_distill_pred, 
                                                                          cur_hs_teacher, 
                                                                          cur_hs_student) 
        if self.segmentation:
            if self.segmentation_stage == 'decoder':
                bs = hs[-1].shape[0]
                Len_in = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum()
                H_level2, W_level2 = spatial_shapes[2]
                memory_rgb = memory[:, :Len_in // 2, :]
                memory_t = memory[:, Len_in // 2:, :]
                memory_rgb_level2 = memory_rgb[:, -(H_level2 * W_level2):, :].permute(0, 2, 1).contiguous().view(bs, -1, H_level2, W_level2)
                memory_t_level2 = memory_t[:, -(H_level2 * W_level2):, :].permute(0, 2, 1).contiguous().view(bs, -1,
                                                                                                                 H_level2,
                                                                                                                 W_level2)

                bbox_mask_rgb = self.bbox_attention_rgb(hs[-1], memory_rgb_level2, mask=features_rgb[-1].decompose()[-1])
                bbox_mask_t = self.bbox_attention_t(hs[-1], memory_t_level2, mask=features_rgb[-1].decompose()[-1])

                seg_masks_rgb = self.mask_head_rgb(srcs_rgb[-1], bbox_mask_rgb, [features_rgb[1].tensors, features_rgb[0].tensors])
                seg_masks_t = self.mask_head_t(srcs_t[-1], bbox_mask_t, [features_rgb[1].tensors, features_rgb[0].tensors])

                outputs_seg_masks_rgb = seg_masks_rgb.view(bs, self.num_queries, seg_masks_rgb.shape[-2], seg_masks_rgb.shape[-1])
                outputs_seg_masks_t = seg_masks_t.view(bs, self.num_queries, seg_masks_t.shape[-2], seg_masks_t.shape[-1])

                out["pred_masks_rgb"] = outputs_seg_masks_rgb
                out["pred_masks_t"] = outputs_seg_masks_t
            else:
                out["pred_masks_rgb"] = pred_masks_rgb
                out["pred_masks_t"] = pred_masks_t
                out["pred_masks_fusion"] = pred_masks_fusion

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out, out_rgb, out_t, out_distill
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


    def _set_aux_loss_4_distill(self, outputs_class, outputs_coord, inter_references_teacher, inter_reference_student, hs_teacher, hs_student):
        return [{'pred_logits': a, 'pred_boxes': b, 'inter_reference_teacher': c, 'inter_reference_student': d, 'hs_teacher': e, 'hs_student': f} 
                    for a, b, c, d, e, f in 
                    zip(outputs_class[:-1], outputs_coord[:-1], inter_references_teacher, inter_reference_student, hs_teacher[:-1], hs_student[:-1])]
  


