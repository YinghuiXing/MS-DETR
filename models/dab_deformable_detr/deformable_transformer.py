# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from util.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn, MSMDDeformRegionAttn
from .racn import Group


class DeformableTransformer(nn.Module):
    def __init__(self, use_dab, rgb_branch, t_branch, modality_crossover, modality_decoupled, two_stage, num_queries, split_cls_reg, cfg, only_fusion, distill, distill_modality_rgb, cfg_distill_decoder, rec_another, rec_flag, rec_momdality_rgb, rec_freeze):
        super().__init__()

        self.two_stage = two_stage
        self.num_queries = num_queries
        self.d_model = cfg.hidden_dim
        self.dim_feedforward = cfg.dim_feedforward
        self.n_heads = cfg.n_heads
        self.use_dab = use_dab
        self.rgb_branch = rgb_branch
        self.t_branch = t_branch

        self.num_feature_levels = cfg.num_feature_levels
        self.key_points_det_share_dec = cfg.DECODER.key_points_det_share_dec
        self.key_points_det_share_modality = cfg.DECODER.key_points_det_share_modality
        self.key_points_det_share_level = cfg.DECODER.key_points_det_share_level
        self.decoder_fusion = cfg.DECODER.fusion
        self.split_cls_reg = split_cls_reg

        self.fusion_backbone_encoder = cfg.fusion_backbone_encoder

        self.distill = distill
        self.distill_modality_rgb = distill_modality_rgb

        self.rec_flag = rec_flag
        self.rec_momdality_rgb = rec_momdality_rgb
        self.rec_freeze = rec_freeze

        encoder_layer = DeformableTransformerEncoderLayer(self.d_model, self.dim_feedforward, self.n_heads, self.num_feature_levels, cfg.ENCODER)
        encoder_norm = nn.LayerNorm(self.d_model) if encoder_layer.pre_norm else None
        level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.d_model))

        self.encoder_rgb = DeformableTransformerEncoder(encoder_layer, cfg.ENCODER.layers_rgb, copy.deepcopy(encoder_norm)) if cfg.ENCODER.layers_rgb else None  # 用于单RGB模态特征
        self.encoder_t = DeformableTransformerEncoder(encoder_layer, cfg.ENCODER.layers_t, copy.deepcopy(encoder_norm)) if cfg.ENCODER.layers_t else None  # 用于单Thermal模态特征
        self.encoder_fusion = DeformableTransformerEncoder(encoder_layer, cfg.ENCODER.layers_fusion, copy.deepcopy(encoder_norm)) if cfg.ENCODER.layers_fusion else None  # 用于融合模态特征
        self.encoder_share = DeformableTransformerEncoder(encoder_layer, cfg.ENCODER.layers_share, copy.deepcopy(encoder_norm)) if cfg.ENCODER.layers_share else None # 两个模态特征共用编码器
        self.encoder_4_fusion = DeformableTransformerEncoder(encoder_layer, cfg.ENCODER.layers_4_fusion, copy.deepcopy(encoder_norm)) if cfg.ENCODER.layers_4_fusion else None # 使用encoder自注意力机制来融合特征

        self.level_embed_rgb = copy.deepcopy(level_embed) if cfg.ENCODER.layers_rgb or cfg.ENCODER.layers_4_fusion else None
        self.level_embed_t = copy.deepcopy(level_embed) if cfg.ENCODER.layers_t or cfg.ENCODER.layers_4_fusion else None  # 如果在encoder中进行特征融合的话，也需要使用单模态的level_embed
        self.level_embed_fusion = copy.deepcopy(level_embed) if cfg.ENCODER.layers_fusion else None
        self.level_embed_share = copy.deepcopy(level_embed) if cfg.ENCODER.layers_share else None

        self.concat_module = AfterEncoderConcatModule(self.d_model) if cfg.after_encoder_concat else None

        decoder_layer = DeformableTransformerDecoderLayer(self.d_model, self.dim_feedforward, self.n_heads, self.num_feature_levels, rgb_branch, t_branch, modality_crossover, modality_decoupled, cfg.fusion_backbone_encoder, only_fusion, cfg.DECODER)
        self.decoder = DeformableTransformerDecoder(decoder_layer, self.use_dab, self.d_model, modality_crossover, only_fusion, split_cls_reg, cfg.DECODER)

        if self.distill:
            decoder_layer_distill = DeformableTransformerDecoderLayer(self.d_model, self.dim_feedforward, self.n_heads, self.num_feature_levels, False, False, False, False, False, False, cfg_distill_decoder)
            self.decoder_distill = DeformableTransformerDecoder(decoder_layer_distill, self.use_dab, self.d_model, False, False, False, cfg_distill_decoder, rec_another=rec_another)
        
        if self.rec_flag:
            self._init_rec_module()

        if cfg.DECODER.use_region_ca:
            self._init_key_point_det_modules(cfg.DECODER.n_points, cfg.DECODER.KDT)

        if two_stage:
            self.enc_output = nn.Linear(self.d_model, self.d_model)
            self.enc_output_norm = nn.LayerNorm(self.d_model)
            self.pos_trans = nn.Linear(self.d_model * 2, self.d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(self.d_model * 2)
        elif not self.use_dab:
            self.reference_points = nn.Linear(self.d_model, 2)

        self._reset_parameters()
        self._init_requires_grad()


    def _init_rec_module(self):
        if self.rec_flag:
            self.rec_module = nn.Sequential(
                nn.Conv1d(self.d_model, self.d_model * 2, 1),
                Group(num_channels=self.d_model * 2, num_blocks=20, reduction=16, use_kernal_3=True),
                nn.Conv1d(self.d_model * 2, self.d_model, 1)
            )
            
    def _init_requires_grad(self):
        if self.distill:
            for name, parameter in self.named_parameters():
                if 'distill' in name:
                    parameter.requires_grad_(True)
                else:
                    parameter.requires_grad_(False)

        if self.rec_flag and self.rec_freeze:
            for name, parameter in self.named_parameters():
                if 'rec_module' in name:
                    parameter.requires_grad_(True)
                else:
                    parameter.requires_grad_(False)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        """
        以下模块拥有定制的参数初始化方法, 因此在对所有参数进行nn.init.xavier_uniform_之后，需要再执行一遍这些模块自己的初始化方法，以保证这
        些定制的参数初始化能够生效
        """
        for m in self.modules():
            if isinstance(m, MSDeformAttn) or isinstance(m, MSMDDeformRegionAttn) or isinstance(m, KeyPointsDet):
                m._reset_parameters()

        if not self.two_stage and not self.use_dab:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)

        if self.level_embed_fusion is not None:
            normal_(self.level_embed_fusion)

        if self.level_embed_rgb is not None:
            normal_(self.level_embed_rgb)

        if self.level_embed_t is not None:
            normal_(self.level_embed_t)

        if self.level_embed_share is not None:
            normal_(self.level_embed_share)

    def _init_key_point_det_4_single(self, dec_n_points, one_layer, trick_init):
        if self.key_points_det_share_level:
            key_point_det = KeyPointsDet(self.d_model, self.n_heads, self.num_feature_levels, dec_n_points, one_layer, trick_init)
        else:
            key_point_det = nn.ModuleList(
                [KeyPointsDet(self.d_model, self.n_heads, 1, dec_n_points, one_layer, trick_init) for _ in range(self.num_feature_levels)]
            )
        return key_point_det

    def _init_key_point_det_modules(self, n_points, cfg):
        dec_n_points = dec_n_points_another = n_points

        # 第一步,为RGB模态所有特征层级生成关键点检测网络
        key_point_det = self._init_key_point_det_4_single(dec_n_points, cfg.one_layer, cfg.trick_init)

        # 第二步, 为Thermal模态生成关键点检测网络
        if self.decoder_fusion:
            if self.key_points_det_share_modality:
                assert dec_n_points_another == dec_n_points
                key_point_det_another = key_point_det
            else:
                key_point_det_another = self._init_key_point_det_4_single(dec_n_points_another, cfg.one_layer, cfg.trick_init)
        else:
            key_point_det_another = None

        # 第三步, 所有decoder层的关键点检测网络
        if self.key_points_det_share_dec:
            for layer in self.decoder.layers:
                layer.cross_attn.key_points_sampling_module = key_point_det
                layer.cross_attn.key_points_sampling_module_another = key_point_det_another
        else:
            for layer in self.decoder.layers:
                layer.cross_attn.key_points_sampling_module = copy.deepcopy(key_point_det)
                layer.cross_attn.key_points_sampling_module_another = copy.deepcopy(key_point_det_another)

    @staticmethod
    def get_proposal_pos_embed(proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes, rgb_flag, t_flag, fusion_flag):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        levels_num = spatial_shapes.shape[0]
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            if rgb_flag and t_flag and lvl == levels_num // 2:
                break

            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)

        if rgb_flag and t_flag:
            proposals = proposals + proposals
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        # for i in range(output_proposals_valid.shape[0]):
        #     for j in range(output_proposals_valid.shape[1]):
        #         if output_proposals_valid[i, j]:
        #             print(output_proposals[i, j])

        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_valid_ratio(mask):
        """
        在DETR中,为了使一个batch中的图像长宽一致，对于较小图像进行填充，该方法用来计算各个图像特征在x和y方向上未填充像素所占比例
        :param mask: Tensor: [bs, h, w]
        :return: Tensor: [bs, 2]
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def _flatten_input(self, srcs, masks, pos_embeds, level_embed):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape

            spatial_shape = (h, w)

            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + level_embed[lvl].view(1, 1, -1)

            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            spatial_shapes.append(spatial_shape)

        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  # [num_levels, 2]
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))  # num_levels
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # bs, num_levels, 2

        return src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten

    def forward(self, input_dict):
        # prepare input for encoder
        rgb_flag = False
        t_flag = False
        fusion_flag = False

        if input_dict['srcs_rgb']:
            rgb_flag = True
            if self.encoder_share:
                encoder_input_rgb = self._flatten_input(input_dict['srcs_rgb'], input_dict['masks_rgb'],
                                                        input_dict['pos_rgb'], self.level_embed_share)
            else:
                encoder_input_rgb = self._flatten_input(input_dict['srcs_rgb'], input_dict['masks_rgb'], input_dict['pos_rgb'], self.level_embed_rgb)

        if input_dict['srcs_t']:
            t_flag = True
            if self.encoder_share:
                encoder_input_t = self._flatten_input(input_dict['srcs_t'], input_dict['masks_t'], input_dict['pos_t'],
                                                      self.level_embed_share)
            else:
                encoder_input_t = self._flatten_input(input_dict['srcs_t'], input_dict['masks_t'], input_dict['pos_t'], self.level_embed_t)

        if input_dict['srcs_fusion']:
            fusion_flag = True
            encoder_input_fusion = self._flatten_input(input_dict['srcs_fusion'], input_dict['masks_fusion'], input_dict['pos_fusion'], self.level_embed_fusion)

        if self.encoder_4_fusion is not None:
            input_4_fusion = list()
            for i, (rgb, t) in enumerate(zip(encoder_input_rgb, encoder_input_t)):
                if i == 1:
                    input_4_fusion.append(torch.cat([rgb, t], dim=0))
                elif i == 2:
                    input_4_fusion.append(torch.cat((input_4_fusion[1].new_zeros((1,)), input_4_fusion[1].prod(1).cumsum(0)[:-1])))
                else:
                    input_4_fusion.append(torch.cat([rgb, t], dim=1))
            memory = self.encoder_4_fusion(*input_4_fusion)           
        else:
            if rgb_flag:
                encoder_rgb = self.encoder_rgb if self.encoder_share is None else self.encoder_share
                memory_rgb = encoder_rgb(*encoder_input_rgb)

            if t_flag:
                encoder_t = self.encoder_t if self.encoder_share is None else self.encoder_share
                memory_t = encoder_t(*encoder_input_t)

        if fusion_flag:
            memory = self.encoder_fusion(*encoder_input_fusion)
            spatial_shapes = encoder_input_fusion[1]
            mask_flatten = encoder_input_fusion[5]
            valid_ratios = encoder_input_fusion[3]
        elif self.concat_module is not None:
            memory = self.concat_module(memory_rgb, memory_t, torch.cat([encoder_input_rgb[1], encoder_input_t[1]], dim=0))
            spatial_shapes = encoder_input_rgb[1]
            mask_flatten = encoder_input_rgb[5]
            valid_ratios = encoder_input_rgb[3]
        elif rgb_flag and t_flag:
            if self.fusion_backbone_encoder:
                memory = torch.cat([memory_rgb, memory_t, encoder_input_rgb[0], encoder_input_t[0]], dim=1)
                spatial_shapes = torch.cat([encoder_input_rgb[1], encoder_input_t[1], encoder_input_rgb[1], encoder_input_t[1]], dim=0)
                mask_flatten = torch.cat([encoder_input_rgb[5], encoder_input_t[5], encoder_input_rgb[5], encoder_input_t[5]], dim=1)
                valid_ratios = torch.cat([encoder_input_rgb[3], encoder_input_t[3], encoder_input_rgb[3], encoder_input_t[3]], dim=1)
            elif self.encoder_4_fusion is not None:
                spatial_shapes = torch.cat([encoder_input_rgb[1], encoder_input_t[1]], dim=0)
                mask_flatten = torch.cat([encoder_input_rgb[5], encoder_input_t[5]], dim=1)
                valid_ratios = torch.cat([encoder_input_rgb[3], encoder_input_t[3]], dim=1)
            else:
                memory = torch.cat([memory_rgb, memory_t], dim=1)
                spatial_shapes = torch.cat([encoder_input_rgb[1], encoder_input_t[1]], dim=0)
                mask_flatten = torch.cat([encoder_input_rgb[5], encoder_input_t[5]], dim=1)
                valid_ratios = torch.cat([encoder_input_rgb[3], encoder_input_t[3]], dim=1)
        elif rgb_flag:
            memory = memory_rgb
            spatial_shapes = encoder_input_rgb[1]
            mask_flatten = encoder_input_rgb[5]
            valid_ratios = encoder_input_rgb[3]
        elif t_flag:
            memory = memory_t
            spatial_shapes = encoder_input_t[1]
            mask_flatten = encoder_input_t[5]
            valid_ratios = encoder_input_t[3]

        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        bs, _, c = memory.shape

        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes, rgb_flag, t_flag, fusion_flag)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)

            tgt_rgb = None
            tgt_t = None
        elif self.use_dab:
            reference_points = input_dict['query_embed'].sigmoid()
            reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points
            tgt = input_dict['tgt_embed'].unsqueeze(0).expand(bs, -1, -1) if input_dict[
                                                                                 'tgt_embed'] is not None else None
            tgt_rgb = input_dict['tgt_embed_rgb'].unsqueeze(0).expand(bs, -1, -1) if input_dict[
                                                                                         'tgt_embed_rgb'] is not None else None
            tgt_t = input_dict['tgt_embed_t'].unsqueeze(0).expand(bs, -1, -1) if input_dict[
                                                                                     'tgt_embed_t'] is not None else None
        else:
            query_embed, tgt = torch.split(input_dict['query_embed'], c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

            if self.distill:
                query_embed_distill, tgt_distill = torch.split(input_dict['query_embed'], c, dim=1)
                query_embed_distill = query_embed_distill.unsqueeze(0).expand(bs, -1, -1)
                tgt_distill = tgt_distill.unsqueeze(0).expand(bs, -1, -1)
                reference_points_distill = self.reference_points(query_embed_distill).sigmoid()
                init_reference_out_distill = reference_points_distill
            else:
                init_reference_out_distill = None

            tgt_rgb = tgt if self.rgb_branch else None
            tgt_t = tgt if self.t_branch else None

            if input_dict['query_embed_rgb'] is not None and input_dict['query_embed_t'] is not None:
                query_embed_rgb, tgt_rgb = torch.split(input_dict['query_embed_rgb'], c, dim=1)
                query_embed_rgb = query_embed_rgb.unsqueeze(0).expand(bs, -1, -1)
                tgt_rgb = tgt_rgb.unsqueeze(0).expand(bs, -1, -1)
                reference_points_rgb = self.reference_points(query_embed_rgb).sigmoid()
                init_reference_out_rgb = reference_points_rgb

                query_embed_t, tgt_t = torch.split(input_dict['query_embed_t'], c, dim=1)
                query_embed_t = query_embed_t.unsqueeze(0).expand(bs, -1, -1)
                tgt_t = tgt_t.unsqueeze(0).expand(bs, -1, -1)
                reference_points_t = self.reference_points(query_embed_t).sigmoid()
                init_reference_out_t = reference_points_t
            else:
                tgt_rgb = tgt if self.rgb_branch else None
                tgt_t = tgt if self.t_branch else None

                query_embed_rgb = None
                reference_points_rgb = None
                init_reference_out_rgb = None

                query_embed_t = None
                reference_points_t = None
                init_reference_out_t = None

        if self.rec_flag:
            if self.rec_momdality_rgb:
                memory_rgb_rec = self.rec_module(memory_t.permute(0, 2, 1)).permute(0, 2, 1)
                memory = torch.cat([memory_rgb_rec, memory_t], dim=1)
            else:
                memory_t_rec = self.rec_module(memory_rgb.permute(0, 2, 1)).permute(0, 2, 1)
                memory = torch.cat([memory_rgb, memory_t_rec], dim=1)

        hs, _, _, hs_rgb, hs_t, \
        inter_references, inter_references_rgb, inter_references_t, inter_references_distill_gt, \
        points_list, weights_list = self.decoder(tgt, tgt_rgb, tgt_t,   
                                    reference_points, reference_points_rgb, reference_points_t,
                                    query_embed if not self.use_dab else None,
                                    query_embed_rgb if not self.use_dab else None,
                                    query_embed_t if not self.use_dab else None,
                                    self.rgb_branch and not rgb_flag,
                                    self.t_branch and not t_flag,
                                    memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten)
        
        if self.distill:
            spatial_shapes_distill = encoder_input_rgb[1] if self.distill_modality_rgb else encoder_input_t[1]
            level_start_index_distill = torch.cat((spatial_shapes_distill.new_zeros((1,)), spatial_shapes_distill.prod(1).cumsum(0)[:-1]))
            valid_ratios_distill = encoder_input_rgb[3] if self.distill_modality_rgb else encoder_input_t[3]
            mask_flatten_distill = encoder_input_rgb[5] if self.distill_modality_rgb else encoder_input_t[5]

            hs_distill, hs_rec_another, hs_distill_before_rec, _, __, \
            inter_references_distill, ___, ____, inter_references_distill_pred, \
            points_list_distill, weights_list_distill = self.decoder_distill(tgt_distill, None, None,
                                        reference_points_distill, None, None,
                                        query_embed_distill, None, None,
                                        False,
                                        False,
                                        memory_rgb if self.distill_modality_rgb else memory_t, 
                                        spatial_shapes_distill, 
                                        level_start_index_distill, 
                                        valid_ratios_distill, 
                                        mask_flatten_distill)
        else:
            hs_distill = None
            inter_references_distill = None
            points_list_distill = None
            weights_list_distill = None
            inter_references_distill_pred = None
            hs_distill_before_rec = None
            hs_rec_another = None

        if self.two_stage:
            return hs, hs_rgb, hs_t, init_reference_out, init_reference_out_rgb, init_reference_out_t, inter_references, inter_references_rgb, inter_references_t, points_list, weights_list, memory, spatial_shapes, enc_outputs_class, enc_outputs_coord_unact
        
        memory_rec = None
        memory_teacher = None
        if self.rec_flag:
            if self.rec_momdality_rgb:
                memory_rec = memory_rgb_rec
                memory_teacher = memory_rgb
            else:
                memory_rec = memory_t_rec
                memory_teacher = memory_t

        return hs, hs_rgb, hs_t, memory_rec, memory_teacher, \
                init_reference_out, init_reference_out_rgb, init_reference_out_t, inter_references_distill_gt,\
                inter_references, inter_references_rgb, inter_references_t, \
                points_list, weights_list, memory, spatial_shapes, None, None, \
                hs_distill, hs_rec_another, hs_distill_before_rec, init_reference_out_distill, inter_references_distill, inter_references_distill_pred, points_list_distill, weights_list_distill


class AfterEncoderConcatModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.feat_1 = nn.Conv2d(d_model * 2, d_model, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_1_bn = nn.BatchNorm2d(d_model, momentum=0.01)

        self.feat_2 = nn.Conv2d(d_model * 2, d_model, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_2_bn = nn.BatchNorm2d(d_model, momentum=0.01)

        self.feat_3 = nn.Conv2d(d_model * 2, d_model, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_3_bn = nn.BatchNorm2d(d_model, momentum=0.01)

        self.feat_4 = nn.Conv2d(d_model * 2, d_model, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_4_bn = nn.BatchNorm2d(d_model, momentum=0.01)

    def forward(self, memory_rgb, memory_t, spatial_shapes):
        H_level_1, W_level_1 = spatial_shapes[0]
        H_level_2, W_level_2 = spatial_shapes[1]
        H_level_3, W_level_3 = spatial_shapes[2]
        H_level_4, W_level_4 = spatial_shapes[3]

        start_ind = 0
        feature_level_1_rgb = memory_rgb[:, start_ind:start_ind + H_level_1 * W_level_1, :].contiguous().view(-1, H_level_1, W_level_1, self.d_model).permute(0, 3, 1, 2)
        feature_level_1_t = memory_t[:, start_ind:start_ind + H_level_1 * W_level_1, :].contiguous().view(-1, H_level_1, W_level_1, self.d_model).permute(0, 3, 1, 2)
        start_ind += H_level_1 * W_level_1
        feature_level_2_rgb = memory_rgb[:, start_ind:start_ind + H_level_2 * W_level_2, :].contiguous().view(-1, H_level_2, W_level_2, self.d_model).permute(0, 3, 1, 2)
        feature_level_2_t = memory_t[:, start_ind:start_ind + H_level_2 * W_level_2, :].contiguous().view(-1, H_level_2, W_level_2, self.d_model).permute(0, 3, 1, 2)
        start_ind += H_level_2 * W_level_2
        feature_level_3_rgb = memory_rgb[:, start_ind:start_ind + H_level_3 * W_level_3, :].contiguous().view(-1, H_level_3, W_level_3, self.d_model).permute(0, 3, 1, 2)
        feature_level_3_t = memory_t[:, start_ind:start_ind + H_level_3 * W_level_3, :].contiguous().view(-1, H_level_3, W_level_3, self.d_model).permute(0, 3, 1, 2)
        start_ind += H_level_3 * W_level_3
        feature_level_4_rgb = memory_rgb[:, start_ind:start_ind + H_level_4 * W_level_4, :].contiguous().view(-1,
                                                                                                              H_level_4,
                                                                                                              W_level_4,
                                                                                                              self.d_model).permute(
            0, 3, 1, 2)
        feature_level_4_t = memory_t[:, start_ind:start_ind + H_level_4 * W_level_4, :].contiguous().view(-1, H_level_4,
                                                                                                          W_level_4,
                                                                                                          self.d_model).permute(
            0, 3, 1, 2)

        layer1_feats = torch.cat([feature_level_1_rgb, feature_level_1_t], dim=1)
        layer1_feats = F.relu(self.feat_1_bn(self.feat_1(layer1_feats))).flatten(2).transpose(1, 2)

        layer2_feats = torch.cat([feature_level_2_rgb, feature_level_2_t], dim=1)
        layer2_feats = F.relu(self.feat_2_bn(self.feat_2(layer2_feats))).flatten(2).transpose(1, 2)

        layer3_feats = torch.cat([feature_level_3_rgb, feature_level_3_t], dim=1)
        layer3_feats = F.relu(self.feat_3_bn(self.feat_3(layer3_feats))).flatten(2).transpose(1, 2)

        layer4_feats = torch.cat([feature_level_4_rgb, feature_level_4_t], dim=1)
        layer4_feats = F.relu(self.feat_4_bn(self.feat_4(layer4_feats))).flatten(2).transpose(1, 2)
        return torch.cat([layer1_feats, layer2_feats, layer3_feats, layer4_feats], dim=1)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, n_heads=8, n_levels=4, cfg=None):
        super().__init__()

        self.pre_norm = cfg.pre_norm

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels if cfg.layers_4_fusion == 0 else n_levels * 2, n_heads, cfg.n_points)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(cfg.activation)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(cfg.dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn_post(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)

        return src

    def forward_ffn_pre(self, src):
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        return src

    def forward_post(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn_post(src)

        return src

    def forward_pre(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.norm1(src)
        src2 = self.self_attn(self.with_pos_embed(src2, pos), reference_points, src2, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)

        src = self.forward_ffn_pre(src)

        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        if self.pre_norm:
            return self.forward_pre(src, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return self.forward_post(src, pos, reference_points, spatial_shapes, level_start_index, padding_mask)


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, encoder_norm):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.encoder_norm = encoder_norm

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - padding_mask: [bs, sum(hi*wi)]
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_lebel, 2]
        """
        output = src
        # bs, sum(hi*wi), 256
        # import ipdb; ipdb.set_trace()
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        if self.encoder_norm is not None:
            output = self.encoder_norm(output)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, n_heads=8, n_levels=4, rgb_branch=False, t_branch=False, modality_crossover=False, modality_decoupled = False, fusion_backbone_encoder=False, only_fusion=False, cfg=None):
        super().__init__()

        self.pre_norm = cfg.pre_norm
        self.rgb_branch = rgb_branch
        self.t_branch = t_branch
        self.branch_share = cfg.branch_share
        self.fusion = cfg.fusion
        self.fusion_backbone_encoder = fusion_backbone_encoder
        self.n_levels = n_levels * 2 if self.fusion else n_levels
        self.n_levels = self.n_levels * 2 if self.fusion_backbone_encoder else self.n_levels
        self.n_points = cfg.n_points
        self.use_sa = cfg.use_sa
        self.modality_crossover = modality_crossover
        self.modality_decoupled = modality_decoupled

        if self.modality_crossover:
            self.branch_share = False

        # cross attention:可以使用两种cross-attention模块
        self.cross_attn = MSMDDeformRegionAttn(d_model, self.n_levels, n_heads, self.n_points, cfg.key_points_det_share_level, self.rgb_branch, self.t_branch, cfg.fusion, cfg.fusion_concat, modality_crossover=self.modality_crossover) if cfg.use_region_ca \
            else MSDeformAttn(d_model, self.n_levels, n_heads, self.n_points, decoder=True, rgb_branch=self.rgb_branch, t_branch=self.t_branch, modality_crossover=self.modality_crossover, modality_decoupled=self.modality_decoupled, only_fusion=only_fusion)
        self.dropout1 = nn.Dropout(cfg.dropout)
        if (cfg.use_region_ca and self.pre_norm) or self.modality_crossover:
            self.norm1 = None
        else:
            self.norm1 = nn.LayerNorm(d_model)

        # self attention
        if self.use_sa and not self.modality_crossover:
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=cfg.dropout)
            self.dropout2 = nn.Dropout(cfg.dropout)
            self.norm2 = nn.LayerNorm(d_model)

        # modality cross attention
        if self.modality_crossover:
            self.modality_cross_attn_rgb = nn.MultiheadAttention(d_model, n_heads, dropout=cfg.dropout)
            self.dropout5_rgb = nn.Dropout(cfg.dropout)
            self.norm4_rgb = nn.LayerNorm(d_model)

            self.modality_cross_attn_t = copy.deepcopy(self.modality_cross_attn_rgb)
            self.dropout5_t = copy.deepcopy(self.dropout5_rgb)
            self.norm4_t = copy.deepcopy(self.norm4_rgb)

        # ffn
        if not self.modality_crossover:
            self.linear1 = nn.Linear(d_model, d_ffn)
            self.activation = _get_activation_fn(cfg.activation)
            self.dropout3 = nn.Dropout(cfg.dropout)
            self.linear2 = nn.Linear(d_ffn, d_model)
            self.dropout4 = nn.Dropout(cfg.dropout)
            self.norm3 = nn.LayerNorm(d_model)

        if (self.rgb_branch and not self.branch_share) or self.modality_crossover:
            # cross attention
            if self.pre_norm and not self.modality_crossover:
                self.norm1_rgb = None
            else:
                self.norm1_rgb = nn.LayerNorm(d_model)

            # self attention
            if self.use_sa:
                self.self_attn_rgb = nn.MultiheadAttention(d_model, n_heads, dropout=cfg.dropout)
                self.norm2_rgb = nn.LayerNorm(d_model)
                self.dropout2_rgb = nn.Dropout(cfg.dropout)

            # ffn
            self.linear1_rgb = nn.Linear(d_model, d_ffn)
            self.activation_rgb = _get_activation_fn(cfg.activation)
            self.dropout3_rgb = nn.Dropout(cfg.dropout)
            self.linear2_rgb = nn.Linear(d_ffn, d_model)
            self.dropout4_rgb = nn.Dropout(cfg.dropout)
            self.norm3_rgb = nn.LayerNorm(d_model)

        if (self.t_branch and not self.branch_share) or self.modality_crossover:
            # cross attention
            if self.pre_norm and not self.modality_crossover:
                self.norm1_t = None
            else:
                self.norm1_t = nn.LayerNorm(d_model)

            # self attention
            if self.use_sa:
                self.self_attn_t = nn.MultiheadAttention(d_model, n_heads, dropout=cfg.dropout)
                self.norm2_t = nn.LayerNorm(d_model)
                self.dropout2_t = nn.Dropout(cfg.dropout)

            # ffn
            self.linear1_t = nn.Linear(d_model, d_ffn)
            self.activation_t = _get_activation_fn(cfg.activation)
            self.dropout3_t = nn.Dropout(cfg.dropout)
            self.linear2_t = nn.Linear(d_ffn, d_model)
            self.dropout4_t = nn.Dropout(cfg.dropout)
            self.norm3_t = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn_post(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward_ffn_post_rgb(self, tgt):
        tgt2 = self.linear2_rgb(self.dropout3(self.activation(self.linear1_rgb(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3_rgb(tgt)

        return tgt

    def forward_ffn_post_t(self, tgt):
        tgt2 = self.linear2_t(self.dropout3(self.activation(self.linear1_t(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3_t(tgt)

        return tgt

    def forward_ffn_pre(self, tgt):
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(tgt2)

        return tgt

    def forward_ffn_pre_rgb(self, tgt):
        tgt2 = self.norm3_rgb(tgt)
        tgt2 = self.linear2_rgb(self.dropout3_rgb(self.activation_rgb(self.linear1_rgb(tgt2))))
        tgt = tgt + self.dropout4_rgb(tgt2)

        return tgt

    def forward_ffn_pre_t(self, tgt):
        tgt2 = self.norm3_t(tgt)
        tgt2 = self.linear2_t(self.dropout3_t(self.activation_t(self.linear1_t(tgt2))))
        tgt = tgt + self.dropout4_t(tgt2)

        return tgt

    def forward_post(self, tgt, tgt_rgb, tgt_t, query_pos, query_pos_rgb, query_pos_t, reference_points, reference_points_rgb, reference_points_t, src, src_spatial_shapes, level_start_index, src_padding_mask):
        # self attention
        if tgt is not None:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        if tgt_rgb is not None:
            q_rgb = k_rgb = self.with_pos_embed(tgt_rgb, query_pos_rgb)

            tgt2_rgb = self.self_attn(q_rgb.transpose(0, 1), k_rgb.transpose(0, 1), tgt_rgb.transpose(0, 1))[0].transpose(0, 1) \
                if self.branch_share \
                else self.self_attn_rgb(q_rgb.transpose(0, 1), k_rgb.transpose(0, 1), tgt_rgb.transpose(0, 1))[0].transpose(0, 1)

            tgt_rgb = tgt_rgb + self.dropout2(tgt2_rgb) if self.branch_share else tgt_rgb + self.dropout2_rgb(tgt2_rgb)
            tgt_rgb = self.norm2(tgt_rgb) if self.branch_share else self.norm2_rgb(tgt_rgb)

        if tgt_t is not None:
            q_t = k_t = self.with_pos_embed(tgt_t, query_pos_t)
            tgt2_t = self.self_attn(q_t.transpose(0, 1), k_t.transpose(0, 1), tgt_t.transpose(0, 1))[0].transpose(0, 1) \
                if self.branch_share \
                else self.self_attn_t(q_t.transpose(0, 1), k_t.transpose(0, 1), tgt_t.transpose(0, 1))[0].transpose(0, 1)
            tgt_t = tgt_t + self.dropout2(tgt2_t) if self.branch_share else tgt_t + self.dropout2_t(tgt2_t)
            tgt_t = self.norm2(tgt_t) if self.branch_share else self.norm2_t(tgt_t)

        # cross attention
        tgt2, points, weights, tgt2_rgb, tgt2_t = self.cross_attn(self.with_pos_embed(tgt, query_pos) if tgt is not None else None,
                                                                  reference_points, src, src_spatial_shapes,
                                                                  level_start_index, src_padding_mask,
                                                                  reference_points_rgb, reference_points_t,
                                                                  self.with_pos_embed(tgt_rgb, query_pos_rgb) if tgt_rgb is not None else None,
                                                                  self.with_pos_embed(tgt_t, query_pos_t) if tgt_t is not None else None)
        if tgt is not None:
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        if tgt_rgb is not None:
            tgt_rgb = tgt_rgb + self.dropout1(tgt2_rgb)
            tgt_rgb = self.norm1(tgt_rgb) if self.branch_share else self.norm1_rgb(tgt_rgb)

        if tgt_t is not None:
            tgt_t = tgt_t + self.dropout1(tgt2_t)
            tgt_t = self.norm1(tgt_t) if self.branch_share else self.norm1_t(tgt_t)

        # ffn
        if tgt is not None:
            tgt = self.forward_ffn_post(tgt)
        if tgt_rgb is not None:
            tgt_rgb = self.forward_ffn_post(tgt_rgb) if self.branch_share else self.forward_ffn_post_rgb(tgt_rgb)
        if tgt_t is not None:
            tgt_t = self.forward_ffn_post(tgt_t) if self.branch_share else self.forward_ffn_post_t(tgt_t)

        return tgt, points, weights, tgt_rgb, tgt_t

    def forward_pre(self, tgt, tgt_rgb, tgt_t, query_pos, query_pos_rgb, query_pos_t, reference_points, reference_points_rgb, reference_points_t, src, src_spatial_shapes, level_start_index, src_padding_mask):
        # self attention
        if tgt is not None and self.use_sa:
            tgt2 = self.norm2(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt2.transpose(0, 1))[0].transpose(0, 1)
            tgt = tgt + self.dropout2(tgt2)

        if tgt_rgb is not None and self.use_sa:
            tgt2_rgb = self.norm2(tgt_rgb) if self.branch_share else self.norm2_rgb(tgt_rgb)
            q_rgb = k_rgb = self.with_pos_embed(tgt2_rgb, query_pos_rgb)
            tgt2_rgb = self.self_attn(q_rgb.transpose(0, 1), k_rgb.transpose(0, 1), tgt2_rgb.transpose(0, 1))[0].transpose(0, 1) \
                if self.branch_share \
                else self.self_attn_rgb(q_rgb.transpose(0, 1), k_rgb.transpose(0, 1), tgt2_rgb.transpose(0, 1))[0].transpose(0, 1)
            tgt_rgb = tgt_rgb + self.dropout2(tgt2_rgb) if self.branch_share else tgt_rgb + self.dropout2_rgb(tgt2_rgb)

        if tgt_t is not None and self.use_sa:
            tgt2_t = self.norm2(tgt_t) if self.branch_share else self.norm2_t(tgt_t)
            q_t = k_t = self.with_pos_embed(tgt2_t, query_pos_t)
            tgt2_t = self.self_attn(q_t.transpose(0, 1), k_t.transpose(0, 1), tgt2_t.transpose(0, 1))[0].transpose(0, 1) \
                if self.branch_share \
                else self.self_attn_t(q_t.transpose(0, 1), k_t.transpose(0, 1), tgt2_t.transpose(0, 1))[0].transpose(0, 1)
            tgt_t = tgt_t + self.dropout2(tgt2_t) if self.branch_share else tgt_t + self.dropout2_t(tgt2_t)

        # 如果是RCCA模块，注意这里没有norm1，因为tgt在ca中不参与计算，如果设置norm1, 这会使norm1的参数无法训练
        tgt2 = self.norm1(tgt) if self.norm1 is not None and tgt is not None else tgt
        tgt2_rgb = self.norm1_rgb(tgt_rgb) if self.modality_crossover else tgt_rgb
        tgt2_t = self.norm1_t(tgt_t) if self.modality_crossover else tgt_t
        tgt2, points, weights, tgt2_rgb, tgt2_t = self.cross_attn(self.with_pos_embed(tgt2, query_pos) if tgt2 is not None else None,
                                                                  reference_points, src, src_spatial_shapes,
                                                                  level_start_index, src_padding_mask,
                                                                  reference_points_rgb, reference_points_t,
                                                                  self.with_pos_embed(tgt2_rgb, query_pos_rgb) if tgt2_rgb is not None else None,
                                                                  self.with_pos_embed(tgt2_t, query_pos_t) if tgt2_t is not None else None)
        if tgt is not None:
            tgt = tgt + self.dropout1(tgt2)

        if tgt_rgb is not None:
            tgt_rgb = tgt_rgb + self.dropout1(tgt2_rgb)

        if tgt_t is not None:
            tgt_t = tgt_t + self.dropout1(tgt2_t)

        # modality cross attention
        if self.modality_crossover:
            temp_tgt_rgb = tgt_rgb.clone()

            tgt2_rgb = self.norm4_rgb(tgt_rgb)
            q = self.with_pos_embed(tgt2_rgb, query_pos_rgb)
            k = self.with_pos_embed(tgt_t, query_pos_t)
            tgt2_rgb = self.modality_cross_attn_rgb(q.transpose(0, 1), k.transpose(0, 1), tgt_t.transpose(0, 1))[0].transpose(0, 1)
            tgt_rgb = tgt_rgb + self.dropout5_rgb(tgt2_rgb)

            tgt2_t = self.norm4_t(tgt_t)
            q = self.with_pos_embed(tgt2_t, query_pos_t)
            k = self.with_pos_embed(temp_tgt_rgb, query_pos_rgb)
            tgt2_t = self.modality_cross_attn_t(q.transpose(0, 1), k.transpose(0, 1), temp_tgt_rgb.transpose(0, 1))[0].transpose(0, 1)
            tgt_t = tgt_t + self.dropout5_t(tgt2_t)

        # ffn
        if tgt is not None:
            tgt = self.forward_ffn_pre(tgt)

        if tgt_rgb is not None:
            tgt_rgb = self.forward_ffn_pre(tgt_rgb) if self.branch_share else self.forward_ffn_pre_rgb(tgt_rgb)

        if tgt_t is not None:
            tgt_t = self.forward_ffn_pre(tgt_t) if self.branch_share else self.forward_ffn_pre_t(tgt_t)

        return tgt, points, weights, tgt_rgb, tgt_t

    def forward(self, tgt, tgt_rgb, tgt_t, query_pos, query_pos_rgb, query_pos_t, reference_points, reference_points_rgb, reference_points_t, src, src_spatial_shapes, level_start_index, src_padding_mask):
        if self.pre_norm:
            return self.forward_pre(tgt, tgt_rgb, tgt_t, query_pos, query_pos_rgb, query_pos_t, reference_points, reference_points_rgb, reference_points_t, src, src_spatial_shapes, level_start_index, src_padding_mask)
        return self.forward_post(tgt, tgt_rgb, tgt_t, query_pos, query_pos_rgb, query_pos_t, reference_points, reference_points_rgb, reference_points_t, src, src_spatial_shapes, level_start_index, src_padding_mask)


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, use_dab, d_model, modality_crossover, only_fusion, split_cls_reg, cfg, rec_another=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, cfg.layers)
        self.modality_crossover = modality_crossover
        self.num_layers = cfg.layers
        self.return_intermediate = cfg.return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR

        self.bbox_embed = None
        self.class_embed = None

        self.bbox_embed_rgb = None
        self.class_embed_rgb = None

        self.bbox_embed_t = None
        self.class_embed_t = None

        self.rec_module = None
        self.rec_another = rec_another

        self.use_dab = use_dab
        self.d_model = d_model
        self.sine_embed = cfg.sine_embed
        self.high_dim_query_update = cfg.high_dim_query_update
        self.only_fusion = only_fusion
        self.split_cls_reg = split_cls_reg

        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 2) if not self.modality_crossover else None
            self.query_scale_rgb = MLP(d_model, d_model, d_model, 2) if self.modality_crossover else None
            self.query_scale_t = MLP(d_model, d_model, d_model, 2) if self.modality_crossover else None

            if self.sine_embed:
                self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
            else:
                self.ref_point_head = MLP(4, d_model, d_model, 3)

        if self.high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, 2)

    def forward(self, tgt, tgt_rgb, tgt_t,
                reference_points, reference_points_rgb, reference_points_t,
                query_pos, query_pos_rgb, query_pos_t,
                rgb_fault, t_fault,
                src, src_spatial_shapes, src_level_start_index, src_valid_ratios, src_padding_mask):
        output = tgt if not rgb_fault and not t_fault else None
        output_rgb = tgt_rgb if not rgb_fault and not self.only_fusion else None
        output_t = tgt_t if not t_fault and not self.only_fusion else None

        # 如果1）没有辅助分支或者2）有辅助分支但是测试时模态数据缺失或者3）辅助分支不会动态更新object_queries时，令该分支的oq为None
        if tgt_rgb is None or rgb_fault or self.bbox_embed_rgb is None or self.modality_crossover or self.only_fusion:
            reference_points_rgb = None
            query_pos_rgb = None
        else:
            if reference_points_rgb is None:
                reference_points_rgb = reference_points
            if query_pos_rgb is None:
                query_pos_rgb = query_pos

        if tgt_t is None or t_fault or self.bbox_embed_t is None or self.modality_crossover or self.only_fusion:
            reference_points_t = None
            query_pos_t = None
        else:
            if reference_points_t is None:
                reference_points_t = reference_points
            if query_pos_t is None:
                query_pos_t = query_pos

        # 得到单模态的src_valid_ratios_rgb
        if reference_points_rgb is not None:
            if t_fault:
                src_valid_ratios_rgb = src_valid_ratios
            else:
                temp_n_levels = src_valid_ratios.shape[1]
                src_valid_ratios_rgb = src_valid_ratios[:, :temp_n_levels // 2, :]

        # 得到单模态的src_valid_ratios_t
        if reference_points_t is not None:
            if rgb_fault:
                src_valid_ratios_t = src_valid_ratios
            else:
                temp_n_levels = src_valid_ratios.shape[1]
                src_valid_ratios_t = src_valid_ratios[:, temp_n_levels // 2:, :]

        intermediate = []
        intermediate_rgb = []
        intermediate_t = []
        intermediate_before_rec = []
        intermediate_rec_another = []

        intermediate_reference_points = []
        intermediate_reference_points_rgb = []
        intermediate_reference_points_t = []

        intermediate_reference_points_4_distill = []

        points_list = []
        weights_list = []

        for lid, layer in enumerate(self.layers):
            """
                用于batch内图像填充的原因，需要对reference points/boxes进行相应调整
                reference_points.shape = [bs, num_queries, 2/4]
                以得到形状为[bs, num_queries, num_levels, 4]
            """
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                             * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None] if not rgb_fault and not t_fault else None
                reference_points_input_rgb = reference_points_rgb[:, :, None] \
                                         * torch.cat([src_valid_ratios_rgb, src_valid_ratios_rgb], -1)[:, None] if reference_points_rgb is not None else None
                reference_points_input_t = reference_points_t[:, :, None] \
                                         * torch.cat([src_valid_ratios_t, src_valid_ratios_t], -1)[:, None] if reference_points_t is not None else None
            elif reference_points.shape[-1] == 2:
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None] if not rgb_fault and not t_fault else None
                reference_points_input_rgb = reference_points_rgb[:, :, None] * src_valid_ratios_rgb[:, None] if reference_points_rgb is not None else None
                reference_points_input_t = reference_points_t[:, :, None] * src_valid_ratios_t[:, None] if reference_points_t is not None else None
            else:
                raise RuntimeError

            if self.use_dab:
                if self.sine_embed:
                    # 根据reference boxes生成position embeddings, 形状为[bs, num_queries, d_model * 2]
                    # 再通过MLP映射为[bs, num_queries, d_model]
                    if self.modality_crossover:
                        query_sine_embed = None
                        query_sine_embed_rgb = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
                        n_levels = reference_points_input.shape[2]
                        query_sine_embed_t = gen_sineembed_for_position(reference_points_input[:, :, n_levels // 2, :])
                    else:
                        query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) if reference_points_input is not None else None
                        query_sine_embed_rgb = gen_sineembed_for_position(reference_points_input_rgb[:, :, 0, :]) if reference_points_input_rgb is not None else None
                        query_sine_embed_t = gen_sineembed_for_position(reference_points_input_t[:, :, 0, :]) if reference_points_input_t is not None else None

                    raw_query_pos = self.ref_point_head(query_sine_embed) if query_sine_embed is not None else None
                    raw_query_pos_rgb = self.ref_point_head(query_sine_embed_rgb) if query_sine_embed_rgb is not None else None
                    raw_query_pos_t = self.ref_point_head(query_sine_embed_t) if query_sine_embed_t is not None else None
                else:
                    raw_query_pos = self.ref_point_head(reference_points_input[:, :, 0, :]) if reference_points_input is not None else None
                    raw_query_pos_rgb = self.ref_point_head(reference_points_input_rgb[:, :, 0, :]) if reference_points_input_rgb is not None else None
                    raw_query_pos_t = self.ref_point_head(reference_points_input_t[:, :, 0, :]) if reference_points_input_t is not None else None

                # 根据上一层decoder的输出结果动态调整当前decoder中object queries的position embeddings
                pos_scale = self.query_scale(output) if lid != 0 and output is not None else 1
                if lid != 0 and output_rgb is not None:
                    if self.modality_crossover:
                        pos_scale_rgb = self.query_scale_rgb(output_rgb)
                    else:
                        pos_scale_rgb = self.query_scale(output_rgb)
                else:
                    pos_scale_rgb = 1
                if lid != 0 and output_t is not None:
                    if self.modality_crossover:
                        pos_scale_t = self.query_scale_t(output_t)
                    else:
                        pos_scale_t = self.query_scale(output_t)
                else:
                    pos_scale_t = 1
                query_pos = pos_scale * raw_query_pos if raw_query_pos is not None else None
                query_pos_rgb = pos_scale_rgb * raw_query_pos_rgb if raw_query_pos_rgb is not None else None
                query_pos_t = pos_scale_t * raw_query_pos_t if raw_query_pos_t is not None else None

            # use_dab == False时, 对position embeddings进行动态调整，默认为False
            if self.high_dim_query_update and lid != 0:
                query_pos = query_pos + self.high_dim_query_proj(output) if output is not None else None
                query_pos_rgb = query_pos_rgb + self.high_dim_query_proj(output_rgb) if output_rgb is not None else None
                query_pos_t = query_pos_t + self.high_dim_query_proj(output_t) if output_t is not None else None

            # 开始进行self-attention, deformable cross-attention以及ffn计算
            output, points, weights, output_rgb, output_t = layer(output, output_rgb, output_t, query_pos, query_pos_rgb, query_pos_t, reference_points_input, reference_points_input_rgb, reference_points_input_t, src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            points_list.append(points)
            weights_list.append(weights)

            if self.rec_module is not None and output is not None:
                output_before_rec = output
                output_after_rec = self.rec_module[lid](output.permute(0, 2, 1)).permute(0, 2, 1)
                if self.rec_another:
                    output_rec_another = output_after_rec
                    output_after_rec = (output_before_rec + output_rec_another) / 2
            else:
                output_before_rec = None
                output_rec_another = None
                output_after_rec = output

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None and output_after_rec is not None:
                if not self.split_cls_reg:
                    tmp = self.bbox_embed[lid](output_after_rec)
                else:
                    tmp = self.bbox_embed[lid](output_after_rec[..., :self.d_model // 2]) # 将分类和回归的特征分开
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                elif reference_points.shape[-1] == 2:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    raise RuntimeError
                # detach() 返回一个新的tensor, 从当前计算图中分离下来，但仍指向原变量的存放位置，不同之处为require_grid为False
                # 永远不需要计算梯度
                reference_points = new_reference_points.detach()

            if self.modality_crossover:
                tmp_rgb = self.bbox_embed_rgb[lid](output_rgb)
                tmp_t = self.bbox_embed_t[lid](output_t)

                output_class_rgb = self.class_embed_rgb[lid](output_rgb)
                output_class_t = self.class_embed_t[lid](output_t)

                pede_logit = torch.stack([output_class_rgb[..., 0], output_class_t[..., 0]], dim=-1)
                tmp_weight = F.softmax(pede_logit, -1)
                tmp_weight_rgb = tmp_weight[..., 0]
                tmp_weight_rgb = tmp_weight_rgb[..., None]
                tmp_weight_t = tmp_weight[..., 1]
                tmp_weight_t = tmp_weight_t[..., None]

                tmp = tmp_rgb * tmp_weight_rgb + tmp_t * tmp_weight_t

                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                elif reference_points.shape[-1] == 2:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    raise RuntimeError
                # detach() 返回一个新的tensor, 从当前计算图中分离下来，但仍指向原变量的存放位置，不同之处为require_grid为False
                # 永远不需要计算梯度
                reference_points = new_reference_points.detach()
            else:
                if self.bbox_embed_rgb is not None and output_rgb is not None:
                    if not self.split_cls_reg:
                        tmp = self.bbox_embed_rgb[lid](output_rgb)
                    else:
                        tmp = self.bbox_embed_rgb[lid](output_rgb[..., :self.d_model // 2]) # 将分类和回归的特征分开
                    if reference_points_rgb.shape[-1] == 4:
                        new_reference_points_rgb = tmp + inverse_sigmoid(reference_points_rgb)
                        new_reference_points_rgb = new_reference_points_rgb.sigmoid()
                    elif reference_points_rgb.shape[-1] == 2:
                        new_reference_points_rgb = tmp
                        new_reference_points_rgb[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points_rgb)
                        new_reference_points_rgb = new_reference_points_rgb.sigmoid()
                    else:
                        print(reference_points_rgb.shape[-1])
                        raise RuntimeError
                    # detach() 返回一个新的tensor, 从当前计算图中分离下来，但仍指向原变量的存放位置，不同之处为require_grid为False
                    # 永远不需要计算梯度
                    reference_points_rgb = new_reference_points_rgb.detach()

                if self.bbox_embed_t is not None and output_t is not None:
                    if not self.split_cls_reg:
                        tmp = self.bbox_embed_t[lid](output_t)
                    else:
                        tmp = self.bbox_embed_t[lid](output_t[..., :self.d_model // 2]) # 将分类和回归的特征分开
                    if reference_points_t.shape[-1] == 4:
                        new_reference_points_t = tmp + inverse_sigmoid(reference_points_t)
                        new_reference_points_t = new_reference_points_t.sigmoid()
                    elif reference_points_t.shape[-1] == 2:
                        new_reference_points_t = tmp
                        new_reference_points_t[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points_t)
                        new_reference_points_t = new_reference_points_t.sigmoid()
                    else:
                        raise RuntimeError
                    # detach() 返回一个新的tensor, 从当前计算图中分离下来，但仍指向原变量的存放位置，不同之处为require_grid为False
                    # 永远不需要计算梯度
                    reference_points_t = new_reference_points_t.detach()

            if self.return_intermediate:
                if output_after_rec is not None:
                    intermediate.append(output_after_rec)
                
                if output_rec_another is not None:
                    intermediate_rec_another.append(output_rec_another)
                
                if output_before_rec is not None:
                    intermediate_before_rec.append(output_before_rec)

                if reference_points is not None:
                    intermediate_reference_points.append(reference_points)
                
                if new_reference_points is not None:
                    intermediate_reference_points_4_distill.append(new_reference_points)

                if output_rgb is not None:
                    intermediate_rgb.append(output_rgb)

                if reference_points_rgb is not None:
                    intermediate_reference_points_rgb.append(reference_points_rgb)

                if output_t is not None:
                    intermediate_t.append(output_t)

                if reference_points_t is not None:
                    intermediate_reference_points_t.append(reference_points_t)

        if self.return_intermediate:
            intermediate = torch.stack(intermediate) if len(intermediate) else None
            intermediate_rgb = torch.stack(intermediate_rgb) if len(intermediate_rgb) else None
            intermediate_t = torch.stack(intermediate_t) if len(intermediate_t) else None
            intermediate_rec_another = torch.stack(intermediate_rec_another) if len(intermediate_rec_another) else None
            intermediate_before_rec = torch.stack(intermediate_before_rec) if len(intermediate_before_rec) else None
            intermediate_reference_points = torch.stack(intermediate_reference_points) if len(intermediate_reference_points) else None
            intermediate_reference_points_rgb = torch.stack(intermediate_reference_points_rgb) if len(intermediate_reference_points_rgb) else None
            intermediate_reference_points_t = torch.stack(intermediate_reference_points_t) if len(intermediate_reference_points_t) else None
            intermediate_reference_points_4_distill = torch.stack(intermediate_reference_points_4_distill) if len(intermediate_reference_points_4_distill) else None
            return intermediate, intermediate_rec_another, intermediate_before_rec, intermediate_rgb, intermediate_t, intermediate_reference_points, intermediate_reference_points_rgb, intermediate_reference_points_t, intermediate_reference_points_4_distill, points_list, weights_list

        return output, output_rgb, output_t, reference_points, points_list, weights_list


class KeyPointsDet(nn.Module):
    def __init__(self, d_model, n_heads, n_levels, n_points, one_layer=False, trick_init=True):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.one_layer = one_layer
        self.trick_init = trick_init

        self.conv_net = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        if one_layer:
            self.sampling_net = nn.Sequential(nn.Linear(self.d_model // 4 * 7 * 7, self.n_heads * self.n_levels * self.n_points * 2))
        else:
            self.sampling_net = nn.Sequential(nn.Linear(self.d_model // 4 * 7 * 7, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.n_heads * self.n_levels * self.n_points * 2)
            )

        if one_layer:
            self.weighting_net = nn.Sequential(nn.Linear(self.d_model // 4 * 7 * 7, self.n_heads * self.n_levels * self.n_points))
        else:
            self.weighting_net = nn.Sequential(
                nn.Linear(self.d_model // 4 * 7 * 7, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.n_heads * self.n_levels * self.n_points),
            )

        self._reset_parameters()

    def _reset_parameters(self):
        # 初始化sampling_net的最后一线性层的参数
        constant_(self.sampling_net[-1].weight.data, 0.)
        if self.trick_init:
            thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)  # 8
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)  # 8, 2
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)  # [n_heads, n_levels, n_points, 2]
            for i in range(self.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                self.sampling_net[-1].bias = nn.Parameter(grid_init.view(-1))
        else:
            nn.init.constant_(self.sampling_net[-1].bias.data, 0)

        # 初始化weighting_net的最后一线性层的参数
        constant_(self.weighting_net[-1].weight.data, 0.)
        constant_(self.weighting_net[-1].bias.data, 0.)

    def forward(self, x):
        N = x.shape[0]  # N == bs * num_queries
        intermediate_features = self.conv_net(x)  # [bs * num_queries, d_model // 4, 7, 7]
        intermediate_features = intermediate_features.reshape(N, self.d_model // 4 * 7 * 7)  # [bs * num_queries, d_model // 4 * 7 * 7]

        points = self.sampling_net(intermediate_features).tanh()  # [bs * num_queries,n_heads * n_levels * n_points * 2]
        weights = self.weighting_net(intermediate_features)  # [bs * num_queries, n_heads * n_levels * n_points]

        return points, weights


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


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 20 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos
