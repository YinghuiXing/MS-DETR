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

import copy
import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, decoder=False, rgb_branch=False, t_branch=False, modality_crossover=False, modality_decoupled=False, only_fusion=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels, it may contain all feature levels of two modalities in RGB-T det
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.decoder = decoder
        self.rgb_branch = rgb_branch
        self.t_branch = t_branch
        self.modality_crossover = modality_crossover
        self.modality_decoupled = modality_decoupled
        self.only_fusion = only_fusion

        if self.modality_crossover or self.modality_decoupled:
            self.sampling_offsets_rgb = nn.Linear(d_model, n_heads * n_levels // 2 * n_points * 2)
            self.sampling_offsets_t = nn.Linear(d_model, n_heads * n_levels // 2 * n_points * 2)

            self.attention_weights_rgb = nn.Linear(d_model, n_heads * n_levels // 2 * n_points)
            self.attention_weights_t = nn.Linear(d_model, n_heads * n_levels // 2 * n_points)

            self.value_proj_rgb = nn.Linear(d_model, d_model)
            self.value_proj_t = nn.Linear(d_model, d_model)

            self.output_proj_rgb = nn.Linear(d_model, d_model)
            self.output_proj_t = nn.Linear(d_model, d_model)

        if not self.modality_crossover:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
            self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
            self.value_proj = nn.Linear(d_model, d_model)
            self.output_proj = nn.Linear(d_model, d_model)

            if self.rgb_branch:
                self.output_proj_rgb = nn.Linear(d_model, d_model)

            if self.t_branch:
                self.output_proj_t = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        if self.modality_crossover or self.modality_decoupled:
            constant_(self.sampling_offsets_rgb.weight.data, 0.)
            constant_(self.sampling_offsets_t.weight.data, 0.)
            grid_init_rgb = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels // 2, self.n_points, 1)
            grid_init_t = copy.deepcopy(grid_init_rgb)
            for i in range(self.n_points):
                grid_init_rgb[:, :, i, :] *= i + 1
                grid_init_t[:, :, i, :] *= i + 1
            with torch.no_grad():
                self.sampling_offsets_rgb.bias = nn.Parameter(grid_init_rgb.view(-1))
                self.sampling_offsets_t.bias = nn.Parameter(grid_init_t.view(-1))
        if not self.modality_crossover:
            constant_(self.sampling_offsets.weight.data, 0.)
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
            for i in range(self.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        if self.modality_crossover or self.modality_decoupled:
            constant_(self.attention_weights_rgb.weight.data, 0.)
            constant_(self.attention_weights_rgb.bias.data, 0.)

            constant_(self.attention_weights_t.weight.data, 0.)
            constant_(self.attention_weights_t.bias.data, 0.)

            xavier_uniform_(self.value_proj_rgb.weight.data)
            constant_(self.value_proj_rgb.bias.data, 0.)
            xavier_uniform_(self.output_proj_rgb.weight.data)
            constant_(self.output_proj_rgb.bias.data, 0.)

            xavier_uniform_(self.value_proj_t.weight.data)
            constant_(self.value_proj_t.bias.data, 0.)
            xavier_uniform_(self.output_proj_t.weight.data)
            constant_(self.output_proj_t.bias.data, 0.)
        if not self.modality_crossover:
            constant_(self.attention_weights.weight.data, 0.)
            constant_(self.attention_weights.bias.data, 0.)
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.)
            xavier_uniform_(self.output_proj.weight.data)
            constant_(self.output_proj.bias.data, 0.)
            if self.rgb_branch:
                xavier_uniform_(self.output_proj_rgb.weight.data)
                constant_(self.output_proj_rgb.bias.data, 0.)
            if self.t_branch:
                xavier_uniform_(self.output_proj_t.weight.data)
                constant_(self.output_proj_t.bias.data, 0.)

    def forward_4_encoder(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
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
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None,
                                 :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None,
                                                                          2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations,
                                            attention_weights, self.im2col_step)
        output = self.output_proj(output)

        return output

    def forward_4_decoder(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, reference_points_rgb=None, reference_points_t=None, query_rgb=None, query_t=None):
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
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)   # 2 * 300 * 8 * 4 * 4 * 2
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        if self.rgb_branch or self.t_branch:
            temp_attention_weights = attention_weights
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                     + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                    'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)

        if self.rgb_branch and not self.only_fusion:
            value_rgb = value[:, :Len_in // 2, :, :].clone()
            input_spatial_shapes_rgb = input_spatial_shapes[:self.n_levels // 2, :]
            input_level_start_index_rgb = input_level_start_index[:self.n_levels // 2]
            sampling_locations_rgb = sampling_locations[:, :, :, :self.n_levels // 2, :, :].clone()
            attention_weights_rgb = F.softmax(temp_attention_weights[..., :self.n_levels * self.n_points // 2], -1).view(N, Len_q, self.n_heads, self.n_levels // 2, self.n_points)

            output_rgb = MSDeformAttnFunction.apply(value_rgb, input_spatial_shapes_rgb, input_level_start_index_rgb, sampling_locations_rgb, attention_weights_rgb, self.im2col_step)
            output_rgb = self.output_proj_rgb(output_rgb)
        else:
            output_rgb = None

        if self.t_branch and not self.only_fusion:
            value_t = value[:, Len_in // 2:, :, :].clone()
            input_spatial_shapes_t = input_spatial_shapes[self.n_levels // 2:, :]
            input_level_start_index_t = input_level_start_index[:self.n_levels // 2]
            sampling_locations_t = sampling_locations[:, :, :, self.n_levels // 2:, :, :].clone()
            attention_weights_t = F.softmax(temp_attention_weights[..., self.n_levels * self.n_points // 2:], -1).view(N, Len_q, self.n_heads, self.n_levels // 2, self.n_points)

            output_t = MSDeformAttnFunction.apply(value_t, input_spatial_shapes_t, input_level_start_index_t, sampling_locations_t, attention_weights_t, self.im2col_step)
            output_t = self.output_proj_t(output_t)
        else:
            output_t = None

        return output, sampling_offsets, attention_weights, output_rgb, output_t

    def forward_4_modality_crossover(self, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, query_rgb=None, query_t=None):
        N, Len_q, _ = query_rgb.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        input_flatten_rgb = input_flatten[:, :Len_in // 2, :]
        input_flatten_t = input_flatten[:, Len_in // 2:, :]

        value_rgb = self.value_proj_rgb(input_flatten_rgb)
        value_t = self.value_proj_t(input_flatten_t)

        if input_padding_mask is not None:
            value_rgb = value_rgb.masked_fill(input_padding_mask[:, :Len_in // 2, None], float(0))
            value_t = value_t.masked_fill(input_padding_mask[:, Len_in // 2:, None], float(0))
        value_rgb = value_rgb.view(N, Len_in // 2, self.n_heads, self.d_model // self.n_heads)
        value_t = value_t.view(N, Len_in // 2, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets_rgb = self.sampling_offsets_rgb(query_rgb).view(N, Len_q, self.n_heads, self.n_levels // 2, self.n_points, 2)
        sampling_offsets_t = self.sampling_offsets_t(query_t).view(N, Len_q, self.n_heads, self.n_levels // 2, self.n_points, 2)
        attention_weights_rgb = self.attention_weights_rgb(query_rgb).view(N, Len_q, self.n_heads, self.n_levels // 2 * self.n_points)
        attention_weights_t = self.attention_weights_t(query_t).view(N, Len_q, self.n_heads, self.n_levels // 2 * self.n_points)

        attention_weights_rgb = F.softmax(attention_weights_rgb, -1).view(N, Len_q, self.n_heads, self.n_levels // 2,
                                                                      self.n_points)
        attention_weights_t = F.softmax(attention_weights_t, -1).view(N, Len_q, self.n_heads, self.n_levels // 2,
                                                                      self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations_rgb = reference_points[:, :, None, :self.n_levels // 2, None, :] \
                                 + sampling_offsets_rgb / offset_normalizer[None, None, None, :self.n_levels // 2, None, :]
            sampling_locations_t = reference_points[:, :, None, self.n_levels // 2:, None, :] \
                                     + sampling_offsets_t / offset_normalizer[None, None, None, self.n_levels // 2:, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations_rgb = reference_points[:, :, None, :self.n_levels // 2, None, :2] + sampling_offsets_rgb / self.n_points * reference_points[:, :, None, :self.n_levels // 2, None, 2:] * 0.5
            sampling_locations_t = reference_points[:, :, None, self.n_levels // 2:, None, :2] + sampling_offsets_t / self.n_points * reference_points[:, :, None, self.n_levels // 2:, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        output_rgb = MSDeformAttnFunction.apply(value_rgb, input_spatial_shapes[:self.n_levels // 2, ...], input_level_start_index[:self.n_levels // 2], sampling_locations_rgb, attention_weights_rgb, self.im2col_step)
        output_t = MSDeformAttnFunction.apply(value_t, input_spatial_shapes[self.n_levels // 2:, ...], input_level_start_index[:self.n_levels // 2], sampling_locations_t, attention_weights_t, self.im2col_step)
        output_rgb = self.output_proj_rgb(output_rgb)
        output_t = self.output_proj_t(output_t)
        return None, None, None, output_rgb, output_t

    def forward_4_modality_decoupled(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, reference_points_rgb=None, reference_points_t=None, query_rgb=None, query_t=None):
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
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        input_flatten_rgb = input_flatten[:, :Len_in // 2, :]
        input_flatten_t = input_flatten[:, Len_in // 2:, :]

        value = self.value_proj(input_flatten)
        value_rgb = self.value_proj_rgb(input_flatten_rgb)
        value_t = self.value_proj_t(input_flatten_t)

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
            value_rgb = value_rgb.masked_fill(input_padding_mask[:, :Len_in // 2, None], float(0))
            value_t = value_t.masked_fill(input_padding_mask[:, Len_in // 2:, None], float(0))

        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        value_rgb = value_rgb.view(N, Len_in // 2, self.n_heads, self.d_model // self.n_heads)
        value_t = value_t.view(N, Len_in // 2, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        sampling_offsets_rgb = self.sampling_offsets_rgb(query_rgb).view(N, Len_q, self.n_heads, self.n_levels // 2, self.n_points, 2)
        sampling_offsets_t = self.sampling_offsets_t(query_t).view(N, Len_q, self.n_heads, self.n_levels // 2, self.n_points, 2)
        attention_weights_rgb = self.attention_weights_rgb(query_rgb).view(N, Len_q, self.n_heads, self.n_levels // 2 * self.n_points)
        attention_weights_t = self.attention_weights_t(query_t).view(N, Len_q, self.n_heads, self.n_levels // 2 * self.n_points)
        attention_weights_rgb = F.softmax(attention_weights_rgb, -1).view(N, Len_q, self.n_heads, self.n_levels // 2, self.n_points)
        attention_weights_t = F.softmax(attention_weights_t, -1).view(N, Len_q, self.n_heads, self.n_levels // 2, self.n_points)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)

            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations_rgb = reference_points_rgb[:, :, None, :, None, :] + sampling_offsets_rgb / offset_normalizer[None, None, None, :self.n_levels // 2, None, :]
            sampling_locations_t = reference_points_t[:, :, None, :, None, :] + sampling_offsets_t / offset_normalizer[None, None, None, self.n_levels // 2:, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            sampling_locations_rgb = reference_points_rgb[:, :, None, :, None, :2] + sampling_offsets_rgb / self.n_points * reference_points_rgb[:, :, None, :, None, 2:] * 0.5
            sampling_locations_t = reference_points_t[:, :, None, :, None, :2] + sampling_offsets_t / self.n_points * reference_points_t[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output_rgb = MSDeformAttnFunction.apply(value_rgb, input_spatial_shapes[:self.n_levels // 2, ...], input_level_start_index[:self.n_levels // 2], sampling_locations_rgb, attention_weights_rgb, self.im2col_step)
        output_t = MSDeformAttnFunction.apply(value_t, input_spatial_shapes[self.n_levels // 2:, ...], input_level_start_index[:self.n_levels // 2], sampling_locations_t, attention_weights_t, self.im2col_step)

        output = self.output_proj(output)
        output_rgb = self.output_proj_rgb(output_rgb)
        output_t = self.output_proj_t(output_t)

        return output, None, None, output_rgb, output_t

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, reference_points_rgb=None, reference_points_t=None, query_rgb=None, query_t=None):
        if self.modality_crossover:
            return self.forward_4_modality_crossover(reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=input_padding_mask, query_rgb=query_rgb, query_t=query_t)
        elif self.modality_decoupled:
            return self.forward_4_modality_decoupled(query, reference_points, input_flatten, input_spatial_shapes,
                                          input_level_start_index, input_padding_mask=input_padding_mask,
                                          reference_points_rgb=reference_points_rgb,
                                          reference_points_t=reference_points_t, query_rgb=query_rgb, query_t=query_t)
        elif self.decoder:
            return self.forward_4_decoder(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=input_padding_mask, reference_points_rgb=reference_points_rgb, reference_points_t=reference_points_t, query_rgb=query_rgb, query_t=query_t)
        else:
            return self.forward_4_encoder(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=input_padding_mask)
