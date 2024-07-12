# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction
from .ms_deform_attn import _is_power_of_2


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class MSMDDeformRegionAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=8, n_heads=8, n_points=4, level_share=False, rgb_branch=False,
                 t_branch=False, fusion=False, fusion_concat=False, modality_crossover=False):
        """
        Multi-Scale&Multi-Modal Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels of all modalities
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads

        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSMDDeformAttn to "
                          "make the dimension of each attention head a power of 2 which is more efficient "
                          "in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.level_share = level_share
        self.rgb_branch = rgb_branch
        self.t_branch = t_branch
        self.fusion = fusion
        self.fusion_concat = fusion_concat

        self.key_points_sampling_module = None
        self.key_points_sampling_module_another = None
        self.region_align_net = None

        self.value_proj = nn.Linear(d_model, d_model)

        if self.fusion_concat:
            self.output_proj = nn.Linear(2 * d_model, d_model)
        else:
            self.output_proj = nn.Linear(d_model, d_model)

        if self.rgb_branch:
            self.output_proj_rgb = nn.Linear(d_model, d_model)

        if self.t_branch:
            self.output_proj_t = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask, reference_points_rgb, reference_points_t, query_rgb, query_t):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        if query is not None:
            Len_q = query.shape[1]
        if query_rgb is not None:
            Len_q = query_rgb.shape[1]
        if query_t is not None:
            Len_q = query_t.shape[1]

        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        rgb_fault = True if query is None and query_rgb is None else False
        t_fault = True if query is None and query_t is None else False

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        if not self.fusion:
            H_level0, W_level0 = input_spatial_shapes[0]
            value_2d_level0 = value[:, :H_level0 * W_level0, :, :].view(N, H_level0, W_level0, self.d_model).permute(0, 3, 1, 2)

            reference_rois_xywh = reference_points[:, :, 0, :]  # [bs, num_queries, 4] - [x_c, y_c, w, h]
            reference_rois = box_cxcywh_to_xyxy(reference_rois_xywh) * torch.as_tensor(
                [W_level0, H_level0, W_level0, H_level0], device=reference_rois_xywh.device)

            roi_features = torchvision.ops.roi_align(
                value_2d_level0,
                list(torch.unbind(reference_rois, dim=0)),
                output_size=(7, 7),
                spatial_scale=1.0,
                aligned=True
            )

            if self.level_share:
                points_relative_box, weights = self.key_points_sampling_module(roi_features)
                points_relative_box_all = points_relative_box.view(N, Len_q, self.n_heads, self.n_levels, -1, 2)
                weights_all = weights.view(N, Len_q, self.n_heads, self.n_levels, -1)
            else:
                points_relative_box_list, weights_list = list(), list()

                for i in range(self.n_levels):
                    points_relative_box, weights = self.key_points_sampling_module[i](roi_features)
                    points_relative_box = points_relative_box.view(N, Len_q, self.n_heads, 1, -1, 2)
                    weights = weights.view(N, Len_q, self.n_heads, 1, -1)

                    points_relative_box_list.append(points_relative_box)
                    weights_list.append(weights)

                points_relative_box_all = torch.cat(points_relative_box_list, dim=3)
                weights_all = torch.cat(weights_list, dim=3)

            points = reference_points[:, :, None, :, None, :2] + points_relative_box_all * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            H_level0, W_level0 = input_spatial_shapes[0]
            if rgb_fault:
                value_2d_level0_rgb = None
                value_2d_level0_t = value[:, Len_in // 2:Len_in // 2 + H_level0 * W_level0, :, :].view(N, H_level0, W_level0, self.d_model).permute(0, 3, 1, 2)
            elif t_fault:
                value_2d_level0_t = None
                value_2d_level0_rgb = value[:, :H_level0 * W_level0, :, :].view(N, H_level0, W_level0, self.d_model).permute(0, 3, 1, 2)
            else:
                value_2d_level0_rgb = value[:, :H_level0 * W_level0, :, :].view(N, H_level0, W_level0, self.d_model).permute(0, 3, 1, 2)
                value_2d_level0_t = value[:, Len_in // 2:Len_in // 2 + H_level0 * W_level0, :, :].view(N, H_level0, W_level0, self.d_model).permute(0, 3, 1, 2)

            assert reference_points_rgb is None
            assert reference_points_t is None
            reference_points_rgb = reference_points[:, :, :self.n_levels // 2, :]
            reference_points_t = reference_points[:, :, self.n_levels // 2:, :]
            reference_rois_xywh = reference_points[:, :, 0, :]  # [bs, num_queries, 4] - [x_c, y_c, w, h]
            reference_rois = box_cxcywh_to_xyxy(reference_rois_xywh) * torch.as_tensor([W_level0, H_level0, W_level0, H_level0], device=reference_rois_xywh.device)
            reference_rois_rgb = reference_rois_t = reference_rois

            roi_features_rgb = torchvision.ops.roi_align(
                value_2d_level0_rgb,
                list(torch.unbind(reference_rois_rgb, dim=0)),
                output_size=(7, 7),
                spatial_scale=1.0,
                aligned=True
            ) if value_2d_level0_rgb is not None else None

            roi_features_t = torchvision.ops.roi_align(
                value_2d_level0_t,
                list(torch.unbind(reference_rois_t, dim=0)),
                output_size=(7, 7),
                spatial_scale=1.0,
                aligned=True
            ) if value_2d_level0_t is not None else None

            if self.level_share:
                points_relative_box, weights = self.key_points_sampling_module(roi_features_rgb) if roi_features_rgb is not None else (None, None)
                points_relative_box_t, weights_t = self.key_points_sampling_module_another(roi_features_t) if roi_features_t is not None else (None, None)

                if points_relative_box is not None:
                    points_relative_box = points_relative_box.view(N, Len_q, self.n_heads, self.n_levels // 2, -1, 2)
                    points = reference_points_rgb[:, :, None, :, None, :2] + points_relative_box * reference_points_rgb[:, :, None, :, None, 2:] * 0.5
                else:
                    points = None

                if points_relative_box_t is not None:
                    points_relative_box_t = points_relative_box_t.view(N, Len_q, self.n_heads, self.n_levels // 2, -1, 2)
                    points_t = reference_points_t[:, :, None, :, None, :2] + points_relative_box_t * reference_points_t[:, :, None, :, None, 2:] * 0.5
                else:
                    points_t = None

                if weights is not None:
                    weights = weights.view(N, Len_q, self.n_heads, self.n_levels // 2, -1)

                if weights_t is not None:
                    weights_t = weights_t.view(N, Len_q, self.n_heads, self.n_levels // 2, -1)

                if points is not None and points_t is not None:
                    points = torch.cat([points, points_t], dim=3)
                elif points is not None:
                    pass
                elif points_t is not None:
                    points = points_t
                else:
                    raise RuntimeError

                if weights is not None and weights_t is not None:
                    weights_all = torch.cat([weights, weights_t], dim=3)
                elif weights is not None:
                    weights_all = weights
                elif weights_t is not None:
                    weights_all = weights_t
                else:
                    raise RuntimeError
            else:
                points_relative_box_list, weights_list = list(), list()
                points_relative_box_list_t, weights_list_t = list(), list()

                for i in range(self.n_levels // 2):
                    points_relative_box, weights = self.key_points_sampling_module[i](roi_features_rgb) if roi_features_rgb is not None else (None, None)
                    points_relative_box_t, weights_t = self.key_points_sampling_module_another[i](roi_features_t) if roi_features_t is not None else (None, None)

                    if points_relative_box is not None:
                        points_relative_box = points_relative_box.view(N, Len_q, self.n_heads, 1, -1, 2)
                    if points_relative_box_t is not None:
                        points_relative_box_t = points_relative_box_t.view(N, Len_q, self.n_heads, 1, -1, 2)

                    if weights is not None:
                        weights = weights.view(N, Len_q, self.n_heads, 1, -1)
                    if weights_t is not None:
                        weights_t = weights_t.view(N, Len_q, self.n_heads, 1, -1)

                    if points_relative_box is not None:
                        points_relative_box_list.append(points_relative_box)
                    if weights is not None:
                        weights_list.append(weights)
                    if points_relative_box_t is not None:
                        points_relative_box_list_t.append(points_relative_box_t)
                    if weights_t is not None:
                        weights_list_t.append(weights_t)

                if len(points_relative_box_list):
                    points = torch.cat(points_relative_box_list, dim=3)
                    points = reference_points_rgb[:, :, None, :, None, :2] + points * reference_points_rgb[:, :, None, :, None, 2:] * 0.5
                else:
                    points = None

                if len(points_relative_box_list_t):
                    points_t = torch.cat(points_relative_box_list_t, dim=3)
                    points_t = reference_points_t[:, :, None, :, None, :2] + points_t * reference_points_t[:, :, None, :, None, 2:] * 0.5
                else:
                    points_t = None

                if points is not None and points_t is not None:
                    points = torch.cat([points, points_t], dim=3)
                elif points is not None:
                    pass
                elif points_t is not None:
                    points = points_t
                else:
                    raise RuntimeError
                weights_list = weights_list + weights_list_t
                weights_all = torch.cat(weights_list, dim=3)

        tmp_weight_all = weights_all.view(N, Len_q, self.n_heads, -1)

        output = None
        output_rgb = None
        output_t = None

        if self.rgb_branch and not rgb_fault:
            attention_weights_rgb = F.softmax(tmp_weight_all[..., :self.n_levels * self.n_points // 2], -1).view(N, Len_q, self.n_heads, self.n_levels // 2, self.n_points)
            value_rgb = value[:, :Len_in // 2, :, :].clone()
            points_rgb = points[:, :, :, :self.n_levels // 2].clone()

            output_rgb = MSDeformAttnFunction.apply(
                value_rgb,
                input_spatial_shapes[:self.n_levels // 2],
                input_level_start_index[:self.n_levels // 2],
                points_rgb,
                attention_weights_rgb,
                self.im2col_step
            )
            output_rgb = self.output_proj_rgb(output_rgb)

        if self.t_branch and not t_fault:
            attention_weights_t = F.softmax(tmp_weight_all[..., self.n_levels * self.n_points // 2:], -1).view(N, Len_q, self.n_heads, self.n_levels // 2, self.n_points) if not rgb_fault else \
                F.softmax(tmp_weight_all, -1).view(N, Len_q, self.n_heads, self.n_levels // 2, self.n_points)
            value_t = value[:, Len_in // 2:, :, :].clone() if not rgb_fault else value.clone()
            points_t = points[:, :, :, self.n_levels // 2:].clone() if not rgb_fault else points.clone()

            output_t = MSDeformAttnFunction.apply(
                value_t,
                input_spatial_shapes[self.n_levels // 2:] if not rgb_fault else input_spatial_shapes,
                input_level_start_index[:self.n_levels // 2] if not rgb_fault else input_level_start_index,
                points_t,
                attention_weights_t,
                self.im2col_step
            )

            output_t = self.output_proj_t(output_t)

        if not rgb_fault and not t_fault:
            if self.fusion_concat:
                output = torch.cat([output_rgb, output_t], dim=-1)
            else:
                attention_weights = F.softmax(tmp_weight_all, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

                output = MSDeformAttnFunction.apply(
                    value,
                    input_spatial_shapes,
                    input_level_start_index,
                    points,
                    attention_weights,
                    self.im2col_step
                )

            output = self.output_proj(output)

        return output, points, F.softmax(tmp_weight_all, -1).view(N, Len_q, self.n_heads, self.n_levels // 2 if rgb_fault or t_fault else self.n_levels, self.n_points), output_rgb, output_t
