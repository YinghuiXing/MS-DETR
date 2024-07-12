# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DModified from eformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, rgb_branch, t_branch, instance_reweight, modality_decoupled, gt_field_class, gt_field_bbox, cfg, reweight_hard, positive_alpha, negative_alpha, adaptive_reweight, plus_three, use_p):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()

        self.cost_class = cfg.set_cost_class
        self.cost_bbox = cfg.set_cost_bbox
        self.cost_giou = cfg.set_cost_giou

        self.rgb_branch = rgb_branch
        self.t_branch = t_branch

        self.gt_field_class = gt_field_class
        self.gt_field_bbox = gt_field_bbox

        # 是否以融合分支的匹配结果作为三个分支的匹配结果，如果为False，则会计算三个分支的匹配结果，然后选择最优（即cost最小）的结果返回
        self.fusion_reference = cfg.fusion_reference
        self.instance_reweight = instance_reweight
        self.reweight_hard = reweight_hard
        self.positive_alpha = positive_alpha
        self.negative_alpha = negative_alpha
        self.adaptive_reweight = adaptive_reweight
        self.modality_decoupled = modality_decoupled
        self.plus_three = plus_three
        self.use_p = use_p

        if self.reweight_hard:
            assert not self.plus_three

    # 计算得到损失矩阵，用做匈牙利匹配的输入
    def _get_cost_matrix(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            # 将预测类别张量的形状由[bs, 100, 类别数]转化为[100 * bs, 类别数]
            # 将预测box张量的形状由[bs, 100, 4]转化为[100 * bs, 4]
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [200, 2]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [200, 4]

            # Also concat the target labels and boxes
            # 假定该batch中总共有7个目标，则得到下述两个张量的形状分别为[7]以及[7, 4]
            tgt_ids = torch.cat([v[self.gt_field_class] for v in targets])  # [7]
            tgt_bbox = torch.cat([v[self.gt_field_bbox] for v in targets])  # [7, 4]

            # Compute the classification cost.
            # 计算分类损失（focal loss）
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]  # [200, 7]

            # Compute the L1 cost between boxes
            # 计算L1损失
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [200, 7]

            # Compute the giou cost betwen boxes
            # 计算giou损失
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))  # [200, 7]

            # Final cost matrix
            # 得到最终的cost
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()  # [2, 100, 7]

            cost_class = cost_class.view(bs, num_queries, -1).cpu()
            cost_bbox = cost_bbox.view(bs, num_queries, -1).cpu()
            cost_giou = cost_giou.view(bs, num_queries, -1).cpu()

            return C, cost_class, cost_bbox, cost_giou

    def forward(self, outputs, targets, outputs_rgb, outputs_t):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            C, C_class, C_bbox, C_giou  = self._get_cost_matrix(outputs, targets)
            if self.rgb_branch and outputs_rgb is not None:
                C_rgb, C_class_rgb, C_bbox_rgb, C_giou_rgb = self._get_cost_matrix(outputs_rgb, targets)
            else:
                C_rgb = None
                C_class_rgb = None
                C_bbox_rgb = None
                C_giou_rgb = None
            if self.t_branch and outputs_t is not None:
                C_t, C_class_t, C_bbox_t, C_giou_t = self._get_cost_matrix(outputs_t, targets)
            else:
                C_t = None
                C_class_t = None
                C_bbox_t = None
                C_giou_t = None

            sizes = [len(v[self.gt_field_bbox]) for v in targets]  # [3, 4]

            # 注意这里的处理，使用split划分成了batch份,再通过c[i]将预测局限于当前batch, 此时c[i]的维度是[num_queries, num_target_boxes_i]
            # 在该例子中分别为[100, 3]和[100, 4]
            # linear_sum_assignment
            # 该列表存储的元素，其数据类型为二维元祖，该列表长度为batch
            # 元祖第一维存储的是被算法认定为正样本的object queries(从小到大顺序)，第二维则存储这些正的object queries对应的target index(无顺序)
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            results = list()
            loss_weights = list()
            loss_class_weights = list()
            loss_bbox_weights = list()
            loss_giou_weights = list()

            if (self.rgb_branch and outputs_rgb is not None) and (self.t_branch and outputs_t is not None):
                indices_rgb = [linear_sum_assignment(c[i]) for i, c in enumerate(C_rgb.split(sizes, -1))] if C_rgb is not None else None
                indices_t = [linear_sum_assignment(c[i]) for i, c in enumerate(C_t.split(sizes, -1))] if C_t is not None else None

                if self.modality_decoupled:
                    results = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j
                               in indices]
                    results_rgb = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j
                               in indices_rgb]
                    results_t = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j
                               in indices_t]
                    return results, None, results_rgb, results_t, None, None, None
                else:
                    # 遍历该batch中的每一张图片
                    for i, (C_matrix, C_class_matrix, C_bbox_matrix, C_giou_matrix, 
                            C_matrix_rgb, C_class_matrix_rgb, C_bbox_matrix_rgb, C_giou_matrix_rgb, 
                            C_matrix_t, C_class_matrix_t, C_bbox_matrix_t, C_giou_matrix_t) \
                    in enumerate(zip(C.split(sizes, -1), C_class.split(sizes, -1), C_bbox.split(sizes, -1), C_giou.split(sizes, -1),
                                    C_rgb.split(sizes, -1), C_class_rgb.split(sizes, -1), C_bbox_rgb.split(sizes, -1), C_giou_rgb.split(sizes, -1),
                                    C_t.split(sizes, -1), C_class_t.split(sizes, -1), C_bbox_t.split(sizes, -1), C_giou_t.split(sizes, -1),)):
                        if not self.fusion_reference:
                            cur_cost = C_matrix[i][indices[i][0], indices[i][1]].sum()
                            cur_cost_rgb = C_matrix_rgb[i][indices_rgb[i][0], indices_rgb[i][1]].sum()
                            cur_cost_t = C_matrix_t[i][indices_t[i][0], indices_t[i][1]].sum()

                            if cur_cost < cur_cost_rgb and cur_cost < cur_cost_t:
                                results.append((torch.as_tensor(indices[i][0], dtype=torch.int64),
                                                torch.as_tensor(indices[i][1], dtype=torch.int64)))
                                cost = C_matrix[i][indices[i][0], indices[i][1]]  # 此处cost的形状为[3]或者[4]
                                cost_class = C_class_matrix[i][indices[i][0], indices[i][1]]
                                cost_bbox = C_bbox_matrix[i][indices[i][0], indices[i][1]]
                                cost_giou = C_giou_matrix[i][indices[i][0], indices[i][1]]

                                cost_rgb = C_matrix_rgb[i][indices[i][0], indices[i][1]]
                                cost_class_rgb = C_class_matrix_rgb[i][indices[i][0], indices[i][1]]
                                cost_bbox_rgb = C_bbox_matrix_rgb[i][indices[i][0], indices[i][1]]
                                cost_giou_rgb = C_giou_matrix_rgb[i][indices[i][0], indices[i][1]]

                                cost_t = C_matrix_t[i][indices[i][0], indices[i][1]]
                                cost_class_t = C_class_matrix_t[i][indices[i][0], indices[i][1]]
                                cost_bbox_t = C_bbox_matrix_t[i][indices[i][0], indices[i][1]]
                                cost_giou_t = C_giou_matrix_t[i][indices[i][0], indices[i][1]]
                            else:
                                if cur_cost_rgb < cur_cost_t:
                                    results.append((torch.as_tensor(indices_rgb[i][0], dtype=torch.int64),
                                                    torch.as_tensor(indices_rgb[i][1], dtype=torch.int64)))

                                    cost = C_matrix[i][indices_rgb[i][0], indices_rgb[i][1]]
                                    cost_class = C_class_matrix[i][indices_rgb[i][0], indices_rgb[i][1]]
                                    cost_bbox = C_bbox_matrix[i][indices_rgb[i][0], indices_rgb[i][1]]
                                    cost_giou = C_giou_matrix[i][indices_rgb[i][0], indices_rgb[i][1]]

                                    cost_rgb = C_matrix_rgb[i][indices_rgb[i][0], indices_rgb[i][1]]
                                    cost_class_rgb = C_class_matrix_rgb[i][indices_rgb[i][0], indices_rgb[i][1]]
                                    cost_bbox_rgb = C_bbox_matrix_rgb[i][indices_rgb[i][0], indices_rgb[i][1]]
                                    cost_giou_rgb = C_giou_matrix_rgb[i][indices_rgb[i][0], indices_rgb[i][1]]

                                    cost_t = C_matrix_t[i][indices_rgb[i][0], indices_rgb[i][1]]
                                    cost_class_t = C_class_matrix_t[i][indices_rgb[i][0], indices_rgb[i][1]]
                                    cost_bbox_t = C_bbox_matrix_t[i][indices_rgb[i][0], indices_rgb[i][1]]
                                    cost_giou_t = C_giou_matrix_t[i][indices_rgb[i][0], indices_rgb[i][1]]
                                else:
                                    results.append((torch.as_tensor(indices_t[i][0], dtype=torch.int64),
                                                    torch.as_tensor(indices_t[i][1], dtype=torch.int64)))

                                    cost = C_matrix[i][indices_t[i][0], indices_t[i][1]]
                                    cost_class = C_class_matrix[i][indices_t[i][0], indices_t[i][1]]
                                    cost_bbox = C_bbox_matrix[i][indices_t[i][0], indices_t[i][1]]
                                    cost_giou = C_giou_matrix[i][indices_t[i][0], indices_t[i][1]]

                                    cost_rgb = C_matrix_rgb[i][indices_t[i][0], indices_t[i][1]]
                                    cost_class_rgb = C_class_matrix_rgb[i][indices_t[i][0], indices_t[i][1]]
                                    cost_bbox_rgb = C_bbox_matrix_rgb[i][indices_t[i][0], indices_t[i][1]]
                                    cost_giou_rgb = C_giou_matrix_rgb[i][indices_t[i][0], indices_t[i][1]]

                                    cost_t = C_matrix_t[i][indices_t[i][0], indices_t[i][1]]
                                    cost_class_t = C_class_matrix_t[i][indices_t[i][0], indices_t[i][1]]
                                    cost_bbox_t = C_bbox_matrix_t[i][indices_t[i][0], indices_t[i][1]]
                                    cost_giou_t = C_giou_matrix_t[i][indices_t[i][0], indices_t[i][1]]
                        else:
                            cost = C_matrix[i][indices[i][0], indices[i][1]]
                            cost_class = C_class_matrix[i][indices[i][0], indices[i][1]]
                            cost_bbox = C_bbox_matrix[i][indices[i][0], indices[i][1]]
                            cost_giou = C_giou_matrix[i][indices[i][0], indices[i][1]]

                            cost_rgb = C_matrix_rgb[i][indices[i][0], indices[i][1]]
                            cost_class_rgb = C_class_matrix_rgb[i][indices[i][0], indices[i][1]]
                            cost_bbox_rgb = C_bbox_matrix_rgb[i][indices[i][0], indices[i][1]]
                            cost_giou_rgb = C_giou_matrix_rgb[i][indices[i][0], indices[i][1]]

                            cost_t = C_matrix_t[i][indices[i][0], indices[i][1]]
                            cost_class_t = C_class_matrix_t[i][indices[i][0], indices[i][1]]
                            cost_bbox_t = C_bbox_matrix_t[i][indices[i][0], indices[i][1]]
                            cost_giou_t = C_giou_matrix_t[i][indices[i][0], indices[i][1]]

                            results = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

                        if self.instance_reweight:
                            cost_4_reweight = cost.detach()
                            cost_class_4_reweight = cost_class.detach()
                            cost_bbox_4_reweight = cost_bbox.detach()
                            cost_giou_4_reweight = cost_giou.detach()

                            cost_rgb_4_reweight = cost_rgb.detach()
                            cost_class_rgb_4_reweight = cost_class_rgb.detach()
                            cost_bbox_rgb_4_reweight = cost_bbox_rgb.detach()
                            cost_giou_rgb_4_reweight = cost_giou_rgb.detach()

                            cost_t_4_reweight = cost_t.detach()
                            cost_class_t_4_reweight = cost_class_t.detach()
                            cost_bbox_t_4_reweight = cost_bbox_t.detach()
                            cost_giou_t_4_reweight = cost_giou_t.detach()

                            cost_all = torch.stack([cost_4_reweight, cost_rgb_4_reweight, cost_t_4_reweight]) # 3 * N, N表示正样本数量
                            cost_class_all = torch.stack([cost_class_4_reweight, cost_class_rgb_4_reweight, cost_class_t_4_reweight])
                            cost_bbox_all = torch.stack([cost_bbox_4_reweight, cost_bbox_rgb_4_reweight, cost_bbox_t_4_reweight])
                            cost_giou_all = torch.stack([cost_giou_4_reweight, cost_giou_rgb_4_reweight, cost_giou_t_4_reweight])

                            cost_all_softmax = F.softmax(cost_all, dim=0) # 3 * N, N表示正样本数量
                            cost_class_all_softmax = F.softmax(cost_class_all, dim=0)
                            cost_bbox_all_softmax = F.softmax(cost_bbox_all, dim=0)
                            cost_giou_all_softmax = F.softmax(cost_giou_all, dim=0)

                            if self.reweight_hard:
                                num_pred = cost_all_softmax.shape[-1]
                                for i in range(num_pred):
                                    cost_fusion_single = cost_all_softmax[0, i]
                                    cost_class_fusion_single = cost_class_all_softmax[0, i]
                                    cost_bbox_fusion_single = cost_bbox_all_softmax[0, i]
                                    cost_giou_fusion_single = cost_giou_all_softmax[0, i]

                                    cost_rgb_single = cost_all_softmax[1, i]
                                    cost_class_rgb_single = cost_class_all_softmax[1, i]
                                    cost_bbox_rgb_single = cost_bbox_all_softmax[1, i]
                                    cost_giou_rgb_single = cost_giou_all_softmax[1, i]

                                    cost_t_single = cost_all_softmax[2, i]
                                    cost_class_t_single = cost_class_all_softmax[2, i]
                                    cost_bbox_t_single = cost_bbox_all_softmax[2, i]
                                    cost_giou_t_single = cost_giou_all_softmax[2, i]

                                    # print(score_fusion + score_rgb + score_t)
                                    cost_min = min(cost_fusion_single, cost_rgb_single, cost_t_single)
                                    cost_class_min = min(cost_class_fusion_single, cost_class_rgb_single, cost_class_t_single)
                                    cost_bbox_min = min(cost_bbox_fusion_single, cost_bbox_rgb_single, cost_bbox_t_single)
                                    cost_giou_min = min(cost_giou_fusion_single, cost_giou_rgb_single, cost_giou_t_single)

                                    p_fusion = cost_fusion_single / cost_min
                                    p_class_fusion = cost_class_fusion_single / cost_class_min
                                    p_bbox_fusion = cost_bbox_fusion_single / cost_bbox_min
                                    p_giou_fusion = cost_giou_fusion_single / cost_giou_min

                                    p_rgb = cost_rgb_single / cost_min
                                    p_class_rgb = cost_class_rgb_single / cost_class_min
                                    p_bbox_rgb = cost_bbox_rgb_single / cost_bbox_min
                                    p_giou_rgb = cost_giou_rgb_single / cost_giou_min

                                    p_t = cost_t_single / cost_min
                                    p_class_t = cost_class_t_single / cost_class_min
                                    p_bbox_t = cost_bbox_t_single / cost_bbox_min
                                    p_giou_t = cost_giou_t_single / cost_giou_min

                                    if self.use_p:
                                        cost_all_softmax[0, i] = p_fusion
                                        cost_all_softmax[1, i] = p_rgb
                                        cost_all_softmax[2, i] = p_t

                                        cost_class_all_softmax[0, i] = p_class_fusion
                                        cost_class_all_softmax[1, i] = p_class_rgb
                                        cost_class_all_softmax[2, i] = p_class_t

                                        cost_bbox_all_softmax[0, i] = p_bbox_fusion
                                        cost_bbox_all_softmax[1, i] = p_bbox_rgb
                                        cost_bbox_all_softmax[2, i] = p_bbox_t
                                        
                                        cost_giou_all_softmax[0, i] = p_giou_fusion
                                        cost_giou_all_softmax[1, i] = p_giou_rgb
                                        cost_giou_all_softmax[2, i] = p_giou_t
                                    else:
                                        cost_all_softmax[0, i] = get_weight_from_p(p_fusion, self.positive_alpha[0], self.negative_alpha[0], self.adaptive_reweight)
                                        cost_class_all_softmax[0, i] = get_weight_from_p(p_class_fusion, self.positive_alpha[0], self.negative_alpha[0], self.adaptive_reweight)
                                        cost_bbox_all_softmax[0, i] = get_weight_from_p(p_bbox_fusion, self.positive_alpha[0], self.negative_alpha[0], self.adaptive_reweight)
                                        cost_giou_all_softmax[0, i] = get_weight_from_p(p_giou_fusion, self.positive_alpha[0], self.negative_alpha[0], self.adaptive_reweight)

                                        cost_all_softmax[1, i] = get_weight_from_p(p_rgb, self.positive_alpha[1], self.negative_alpha[1], self.adaptive_reweight)
                                        cost_class_all_softmax[1, i] = get_weight_from_p(p_class_rgb, self.positive_alpha[1], self.negative_alpha[1], self.adaptive_reweight)
                                        cost_bbox_all_softmax[1, i] = get_weight_from_p(p_bbox_rgb, self.positive_alpha[1], self.negative_alpha[1], self.adaptive_reweight)
                                        cost_giou_all_softmax[1, i] = get_weight_from_p(p_giou_rgb, self.positive_alpha[1], self.negative_alpha[1], self.adaptive_reweight)

                                        cost_all_softmax[2, i] = get_weight_from_p(p_t, self.positive_alpha[2], self.negative_alpha[2], self.adaptive_reweight)
                                        cost_class_all_softmax[2, i] = get_weight_from_p(p_class_t, self.positive_alpha[2], self.negative_alpha[2], self.adaptive_reweight)
                                        cost_bbox_all_softmax[2, i] = get_weight_from_p(p_bbox_t, self.positive_alpha[2], self.negative_alpha[2], self.adaptive_reweight)
                                        cost_giou_all_softmax[2, i] = get_weight_from_p(p_giou_t, self.positive_alpha[2], self.negative_alpha[2], self.adaptive_reweight)

                            loss_weight = cost_all_softmax * 3 if self.plus_three else cost_all_softmax
                            loss_class_weight = cost_class_all_softmax * 3 if self.plus_three else cost_class_all_softmax
                            loss_bbox_weight = cost_bbox_all_softmax * 3 if self.plus_three else cost_bbox_all_softmax
                            loss_giou_weight = cost_giou_all_softmax * 3 if self.plus_three else cost_giou_all_softmax
                        else:
                            cost_4_reweight = cost.sum().detach()
                            cost_class_4_reweight = cost_class.sum().detach()
                            cost_bbox_4_reweight = cost_bbox.sum().detach()
                            cost_giou_4_reweight = cost_giou.sum().detach()

                            cost_rgb_4_reweight = cost_rgb.sum().detach()
                            cost_class_rgb_4_reweight = cost_class_rgb.sum().detach()
                            cost_bbox_rgb_4_reweight = cost_bbox_rgb.sum().detach()
                            cost_giou_rgb_4_reweight = cost_giou_rgb.sum().detach()

                            cost_t_4_reweight = cost_t.sum().detach()
                            cost_class_t_4_reweight = cost_class_t.sum().detach()
                            cost_bbox_t_4_reweight = cost_bbox_t.sum().detach()
                            cost_giou_t_4_reweight = cost_giou_t.sum().detach()

                            loss_weight = F.softmax(torch.as_tensor([cost_4_reweight, cost_rgb_4_reweight, cost_t_4_reweight], dtype=torch.float32), -1)  # loss_weight的形状为3
                            loss_class_weight = F.softmax(torch.as_tensor([cost_class_4_reweight, cost_class_rgb_4_reweight, cost_class_t_4_reweight], dtype=torch.float32), -1)
                            loss_bbox_weight = F.softmax(torch.as_tensor([cost_bbox_4_reweight, cost_bbox_rgb_4_reweight, cost_bbox_t_4_reweight], dtype=torch.float32), -1)
                            loss_giou_weight = F.softmax(torch.as_tensor([cost_giou_4_reweight, cost_giou_rgb_4_reweight, cost_giou_t_4_reweight], dtype=torch.float32), -1)

                            if self.reweight_hard:
                                l_fusion = loss_weight[0]
                                l_rgb = loss_weight[1]
                                l_t = loss_weight[2]
                                l_min = min(l_fusion, l_rgb, l_t)

                                loss_weight[0] = get_weight_from_p(l_fusion / l_min, self.positive_alpha[0], self.negative_alpha[0], self.adaptive_reweight)
                                loss_weight[1] = get_weight_from_p(l_rgb / l_min, self.positive_alpha[1], self.negative_alpha[1], self.adaptive_reweight)
                                loss_weight[2] = get_weight_from_p(l_t / l_min, self.positive_alpha[2], self.negative_alpha[2], self.adaptive_reweight)

                                l_fusion_class = loss_class_weight[0]
                                l_rgb_class = loss_class_weight[1]
                                l_t_class = loss_class_weight[2]
                                l_min_class = min(l_fusion_class, l_rgb_class, l_t_class)

                                loss_class_weight[0] = get_weight_from_p(l_fusion_class / l_min_class, self.positive_alpha[0], self.negative_alpha[0], self.adaptive_reweight)
                                loss_class_weight[1] = get_weight_from_p(l_rgb_class / l_min_class, self.positive_alpha[1], self.negative_alpha[1], self.adaptive_reweight)
                                loss_class_weight[2] = get_weight_from_p(l_t_class / l_min_class, self.positive_alpha[2], self.negative_alpha[2], self.adaptive_reweight)

                                l_fusion_bbox = loss_bbox_weight[0]
                                l_rgb_bbox = loss_bbox_weight[1]
                                l_t_bbox = loss_bbox_weight[2]
                                l_min_bbox = min(l_fusion_bbox, l_rgb_bbox, l_t_bbox)

                                loss_bbox_weight[0] = get_weight_from_p(l_fusion_bbox / l_min_bbox, self.positive_alpha[0], self.negative_alpha[0], self.adaptive_reweight)
                                loss_bbox_weight[1] = get_weight_from_p(l_rgb_bbox / l_min_bbox, self.positive_alpha[1], self.negative_alpha[1], self.adaptive_reweight)
                                loss_bbox_weight[2] = get_weight_from_p(l_min_bbox / l_min_bbox, self.positive_alpha[2], self.negative_alpha[2], self.adaptive_reweight)

                                l_fusion_giou = loss_giou_weight[0]
                                l_rgb_giou = loss_giou_weight[1]
                                l_t_giou = loss_giou_weight[2]
                                l_min_giou = min(l_fusion_giou, l_rgb_giou, l_t_giou)

                                loss_giou_weight[0] = get_weight_from_p(l_fusion_giou / l_min_giou, self.positive_alpha[0], self.negative_alpha[0], self.adaptive_reweight)
                                loss_giou_weight[1] = get_weight_from_p(l_rgb_giou / l_min_giou, self.positive_alpha[1], self.negative_alpha[1], self.adaptive_reweight)
                                loss_giou_weight[2] = get_weight_from_p(l_t_giou / l_min_giou, self.positive_alpha[2], self.negative_alpha[2], self.adaptive_reweight)

                            if self.plus_three:
                                loss_weight = loss_weight * 3
                                loss_class_weight = loss_class_weight * 3
                                loss_bbox_weight = loss_bbox_weight * 3
                                loss_giou_weight = loss_giou_weight * 3

                        loss_weights.append(loss_weight)
                        loss_class_weights.append(loss_class_weight)
                        loss_bbox_weights.append(loss_bbox_weight)
                        loss_giou_weights.append(loss_giou_weight)
                    if self.instance_reweight:
                        loss_weights = torch.cat(loss_weights, dim=-1)  # [3, total_gt_of_batch]
                        loss_class_weights = torch.cat(loss_class_weights, dim=-1)  # [3, total_gt_of_batch]
                        loss_bbox_weights = torch.cat(loss_bbox_weights, dim=-1)  # [3, total_gt_of_batch]
                        loss_giou_weights = torch.cat(loss_giou_weights, dim=-1)  # [3, total_gt_of_batch]
                    else:
                        loss_weights = torch.stack(loss_weights, dim=0)  # [bs, 3]
                        loss_class_weights = torch.stack(loss_class_weights, dim=0)
                        loss_bbox_weights = torch.stack(loss_bbox_weights, dim=0)
                        loss_giou_weights = torch.stack(loss_giou_weights, dim=0)
                    return results, loss_weights, None, None, loss_class_weights, loss_bbox_weights, loss_giou_weights
            else:
                results = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
                return results, None, None, None, None, None, None


def get_weight_from_p(p, positive_alpha = 1.0, negative_alpha = 0.0, adaptive_reweight=False):
    if p == 1.0:
        return negative_alpha
    else:
        if not adaptive_reweight:
            return positive_alpha
        
        weight = 1.0 if p - 1 > 1.0 else p - 1
        weight = negative_alpha + weight
        return max(weight, 0.0)

