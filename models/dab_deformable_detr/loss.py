# -*- coding: utf-8 -*-
# @Time    : 2023/2/12 15:12
# @Author  : WangSong
# @Email   : 1021707198@qq.com
# @File    : loss.py

import os
import torch
import copy
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (accuracy, get_world_size, is_dist_avail_and_initialized, interpolate)

from .segmentation import sigmoid_focal_loss, dice_loss
from util.misc import nested_tensor_from_tensor_list

def EU_dist(x1, x2):
    d_matrix = torch.zeros(x1.shape[0], x2.shape[0]).to(x1.device)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            d = torch.sqrt(torch.dot((x1[i] - x2[j]), (x1[i] - x2[j])))
            d_matrix[i, j] = d
    return d_matrix

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, gt_field_class, gt_field_bbox, focal_alpha=0.25, instance_reweight=False, term_reweight=False, 
                 prototype=False, prototype_reweight=False, prototype_all_layers=True, prototype_alpha=1.0, follow_last_layer=False, split_cls_reg=False, adaptive_reweight=False, 
                 distill=False, follow_teacher=False, distill_features_loss='mse', rec_features_loss='mse'):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.gt_field_class = gt_field_class  # 多模态中选择哪一个模态的标签进行训练
        self.gt_field_bbox = gt_field_bbox  # 多模态中选择哪一个模态的标签进行训练
        self.focal_alpha = focal_alpha
        self.instance_reweight = instance_reweight
        self.term_reweight = term_reweight
        self.follow_last_layer = follow_last_layer
        self.prototype = prototype
        self.prototype_reweight = prototype_reweight
        self.prototype_all_layers = prototype_all_layers
        self.prototype_alpha = prototype_alpha
        self.split_cls_reg = split_cls_reg
        self.adaptive_reweight = adaptive_reweight
        self.distill = distill
        self.follow_teacher = follow_teacher
        self.distill_features_loss = distill_features_loss
        self.rec_features_loss = rec_features_loss

    def loss_labels(self, outputs, targets, indices, indices_teacher, num_boxes, loss_weight, loss_class_weight, loss_bbox_weight, loss_giou_weight, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[self.gt_field_class][J] for t, (_, J) in zip(targets, indices)])  # [7]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)  # [2, 100]
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]  # [2, 100, 2]

        if self.term_reweight:
            loss_weight = loss_class_weight

        if loss_weight is not None:
            loss_weight = loss_weight.to(src_logits.device)
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2, return_batch_loss=True)
            if self.instance_reweight:
                sizes = [len(v[self.gt_field_bbox]) for v in targets]
                loss_weight_batch = loss_weight.split(sizes)
                batch_ind = idx[0].split(sizes)
                query_ind = idx[1].split(sizes)

                for a, b, c in zip(loss_weight_batch, batch_ind, query_ind):
                    loss_ce[b, c] *= a[:, None]
            else:
                loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2, return_batch_loss=True) * loss_weight[:, None, None]
            loss_ce = loss_ce.mean(1).sum() / num_boxes * src_logits.shape[1]
        else:
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, indices_teacher, num_boxes, loss_weight, loss_class_weight, loss_bbox_weight, loss_giou_weight):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v[self.gt_field_class]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, indices_teacher, num_boxes, loss_weight, loss_class_weight, loss_bbox_weight, loss_giou_weight):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t[self.gt_field_bbox][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        if self.term_reweight:
            assert loss_bbox_weight is not None
            loss_bbox_weight = loss_bbox_weight.to(src_boxes.device)
            sizes = [len(v[self.gt_field_bbox]) for v in targets]
            loss_bbox *= loss_bbox_weight[:, None]
        elif loss_weight is not None:
            loss_weight = loss_weight.to(src_boxes.device)
            sizes = [len(v[self.gt_field_bbox]) for v in targets]

            if self.instance_reweight:
                loss_bbox *= loss_weight[:, None]
            else:
                loss_bbox = torch.cat([(l * loss_weight[bs]) for bs, l in enumerate(loss_bbox.split(sizes))])

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        if self.term_reweight:
            loss_giou *= loss_giou_weight.to(src_boxes.device)
        elif loss_weight is not None:
            if self.instance_reweight:
                loss_giou *= loss_weight
            else:
                loss_giou = torch.cat([(l * loss_weight[bs]) for bs, l in enumerate(loss_giou.split(sizes))])

        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def loss_distill_features(self, outputs, targets, indices, indices_teacher, num_boxes, loss_weight, loss_class_weight, loss_bbox_weight, loss_giou_weight):
        assert 'hs_teacher' in outputs
        assert 'hs_student' in outputs

        # print(indices)
        # print(indices_teacher)
        indices_after_order = self._order_indices_by_gt(indices)
        indices_teacher_after_order = self._order_indices_by_gt(indices_teacher)
        idx = self._get_src_permutation_idx(indices_after_order)
        idx_teacher = self._get_src_permutation_idx(indices_teacher_after_order)

        # print(indices_after_order)
        # print(indices_teacher_after_order)
        # print(idx)
        # print(idx_teacher)

        src_features = outputs['hs_student'][idx]
        target_features = outputs['hs_teacher'][idx_teacher].detach()
        
        if self.distill_features_loss == 'mse':
            src_features = nn.functional.normalize(src_features, dim=1)
            target_features = nn.functional.normalize(target_features, dim=1)

            loss_distill_features = F.mse_loss(src_features, target_features, reduction='none')
        elif self.distill_features_loss == 'L1':
            loss_distill_features = F.l1_loss(src_features, target_features, reduction='none')

        losses = {}
        losses['loss_distill_features'] = loss_distill_features.sum() / num_boxes

        return losses
    
    def loss_rec_features(self, outputs, targets, indices, indices_teacher, num_boxes, loss_weight, loss_class_weight, loss_bbox_weight, loss_giou_weight):
        assert 'memory_rec' in outputs
        assert 'memory_teacher' in outputs

        if self.rec_features_loss == 'mse':
            src_features = nn.functional.normalize(outputs['memory_rec'], dim=2)
            target_features = nn.functional.normalize(outputs['memory_teacher'], dim=2)

            loss_distill_features = F.mse_loss(src_features, target_features, reduction='none')
        elif self.rec_features_loss == 'L1':
            loss_distill_features = F.l1_loss(outputs['memory_rec'], outputs['memory_teacher'], reduction='none')

        losses = {}
        losses['loss_rec_features'] = loss_distill_features.sum() / num_boxes

        return losses

    def loss_distill_boxes(self, outputs, targets, indices, indices_teacher, num_boxes, loss_weight, loss_class_weight, loss_bbox_weight, loss_giou_weight):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'inter_reference_teacher' in outputs
        assert 'inter_reference_student' in outputs

        indices_after_order = self._order_indices_by_gt(indices)
        indices_teacher_after_order = self._order_indices_by_gt(indices_teacher)
        idx = self._get_src_permutation_idx(indices_after_order)
        idx_teacher = self._get_src_permutation_idx(indices_teacher_after_order)

        src_boxes = outputs['inter_reference_student'][idx]
        target_boxes = outputs['inter_reference_teacher'][idx_teacher]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_distill_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_distill_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, indices_teacher, num_boxes, loss_weight):
        assert "pred_masks_rgb" in outputs
        assert "pred_masks_t" in outputs
        assert "pred_masks_fusion" in outputs

        src_masks_rgb = outputs["pred_masks_rgb"]
        src_masks_t = outputs["pred_masks_t"]
        src_masks_fusion = outputs["pred_masks_fusion"]
        masks = [t["masks"][None] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        loss_mask_rgb = 0
        loss_mask_t = 0
        loss_mask_fusion = 0
        loss = dict()

        if len(src_masks_rgb):
            for src_mask_rgb in src_masks_rgb:
                target_masks_rgb = target_masks.to(src_mask_rgb)
                target_masks_rgb = interpolate(target_masks_rgb, size=src_mask_rgb.shape[-2:], mode="nearest")
                loss_mask_rgb += dice_loss(src_mask_rgb.flatten(1), target_masks_rgb.flatten(1), 1)
                loss_mask_rgb += F.binary_cross_entropy(src_mask_rgb.sigmoid(), target_masks_rgb)
            loss['loss_mask_rgb'] = loss_mask_rgb

        if len(src_masks_t):
            for src_mask_t in src_masks_t:
                target_masks_t = target_masks.to(src_mask_t)
                target_masks_t = interpolate(target_masks_t, size=src_mask_t.shape[-2:], mode="nearest")
                loss_mask_t += dice_loss(src_mask_t.flatten(1), target_masks_t.flatten(1), 1)
                loss_mask_t += F.binary_cross_entropy(src_mask_t.sigmoid(), target_masks_t)
            loss['loss_mask_t'] = loss_mask_t

        if len(src_masks_fusion):
            for src_mask_fusion in src_masks_fusion:
                target_masks_fusion = target_masks.to(src_mask_fusion)
                target_masks_fusion = interpolate(target_masks_fusion, size=src_mask_fusion.shape[-2:], mode="nearest")
                loss_mask_fusion += dice_loss(src_mask_fusion.flatten(1), target_masks_fusion.flatten(1), 1)
                loss_mask_fusion += F.binary_cross_entropy(src_mask_fusion.sigmoid(), target_masks_fusion)
            loss['loss_mask_fusion'] = loss_mask_fusion

        return loss

    def loss_masks_decoder(self, outputs, targets, indices, indices_teacher, num_boxes, loss_weight):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks_rgb" in outputs
        assert "pred_masks_t" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks_rgb = outputs["pred_masks_rgb"]
        src_masks_t = outputs["pred_masks_t"]
        src_masks_rgb = src_masks_rgb[src_idx]
        src_masks_t = src_masks_t[src_idx]
        masks = [t["masks_decoder"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks_rgb)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks_rgb = interpolate(src_masks_rgb[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks_rgb = src_masks_rgb[:, 0].flatten(1)

        src_masks_t = interpolate(src_masks_t[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks_t = src_masks_t[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks_rgb.shape)
        losses = {
            "loss_mask_rgb": sigmoid_focal_loss(src_masks_rgb, target_masks, num_boxes),
            "loss_dice_rgb": dice_loss(src_masks_rgb, target_masks, num_boxes),
            "loss_mask_t": sigmoid_focal_loss(src_masks_t, target_masks, num_boxes),
            "loss_dice_t": dice_loss(src_masks_t, target_masks, num_boxes),
        }
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @staticmethod
    def _order_indices_by_gt(indices):
        for i, tuple_permutation in enumerate(indices):
            pred_permutation = tuple_permutation[0]
            gt_permutation = tuple_permutation[1]

            sorted_indices = torch.argsort(gt_permutation)
            sorted_pred_permutation = pred_permutation[sorted_indices]
            sorted_gt_permutation = torch.sort(gt_permutation).values
            
            indices[i] = (sorted_pred_permutation, sorted_gt_permutation)
        
        return indices

    def get_loss(self, loss, outputs, targets, indices, indices_teacher, num_boxes, loss_weight, loss_class_weight, loss_bbox_weight, loss_giou_weight, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks_decoder': self.loss_masks_decoder,
            'masks': self.loss_masks,
            'distill_boxes': self.loss_distill_boxes,
            'distill_features': self.loss_distill_features,
            'rec_features': self.loss_rec_features,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, indices_teacher, num_boxes, loss_weight, loss_class_weight, loss_bbox_weight, loss_giou_weight, **kwargs)

    def forward_single(self, outputs, targets, indices, indices_teacher, loss_weight, loss_class_weight, loss_bbox_weight, loss_giou_weight, aux_indices_list, aux_indices_list_teacher, aux_loss_weights_list, aux_loss_class_weights_list, aux_loss_bbox_weights_list, aux_loss_giou_weights_list, fusion_branch):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t[self.gt_field_class]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == 'rec_features' and not fusion_branch:
                continue

            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, indices_teacher, num_boxes, loss_weight, loss_class_weight, loss_bbox_weight, loss_giou_weight, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = aux_indices_list[i]
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if loss == 'illumination':
                        continue
                    if loss == 'rec_features':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    if self.follow_last_layer:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, indices_teacher, num_boxes, loss_weight, loss_class_weight, loss_bbox_weight, loss_giou_weight, **kwargs)
                    else:
                        if aux_loss_weights_list is not None:
                            l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices_list[i], aux_indices_list_teacher[i] if aux_indices_list_teacher is not None else None, num_boxes, aux_loss_weights_list[i], 
                                                   aux_loss_class_weights_list[i] if self.term_reweight else None, 
                                                   aux_loss_bbox_weights_list[i] if self.term_reweight else None, 
                                                   aux_loss_giou_weights_list[i]if self.term_reweight else None, **kwargs)
                        else:
                            l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices_list[i], aux_indices_list_teacher[i] if aux_indices_list_teacher is not None else None, num_boxes, None, None, None, None, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            # for bt in bin_targets:
            #     bt['labels'] = torch.zeros_like(bt['labels'])
            if os.environ.get('IPDB_SHILONG_DEBUG') == 'INFO':
                import ipdb; ipdb.set_trace()
            indices, _ = self.matcher(enc_outputs, bin_targets, None, None)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if loss == 'illumination':
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, None, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def forward(self, outputs, outputs_rgb, outputs_t, outputs_distill, targets, dynamic_weight, rgb_prototypes, t_prototypes, fusion_prototypes):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        if outputs_rgb is not None and outputs_t is not None:
            outputs_rgb_without_aux = {k: v for k, v in outputs_rgb.items() if k != 'aux_outputs' and k != 'enc_outputs'}
            outputs_t_without_aux = {k: v for k, v in outputs_t.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        else:
            outputs_rgb_without_aux = None
            outputs_t_without_aux = None
        
        if outputs_distill is not None:
            outputs_distill_without_aux = {k: v for k, v in outputs_distill.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        else:
            outputs_distill_without_aux = None

        # Retrieve the matching between the outputs of the last layer and the targets
        # 根据Transformer Decoder最后一层的输出得到真值目标与oq之间的对应关系
        if self.distill:
            if self.follow_teacher:
                indices, loss_weights, indices_rgb, indices_t, loss_class_weights, loss_bbox_weights, loss_giou_weights = self.matcher(outputs_without_aux, targets, outputs_rgb_without_aux, outputs_t_without_aux)
                indices_teacher = indices
            else:
                indices, loss_weights, indices_rgb, indices_t, loss_class_weights, loss_bbox_weights, loss_giou_weights = self.matcher(outputs_distill_without_aux, targets, None, None)
                indices_teacher, loss_weights_teacher, indices_rgb_teacher, indices_t_teacher, loss_class_weights_teacher, loss_bbox_weights_teacher, loss_giou_weightsteacher = self.matcher(outputs_without_aux, targets, outputs_rgb_without_aux, outputs_t_without_aux)
        else:
            # 举个例子：indices : [(tensor([ 97, 151, 161, 179]), tensor([3, 0, 1, 2])), (tensor([ 46, 101, 109, 114, 127, 170, 232, 283]), tensor([1, 6, 7, 0, 3, 5, 2, 4]))]
            # self._get_src_permutation_idx(indices) : (tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]), tensor([ 97, 151, 161, 179,  46, 101, 109, 114, 127, 170, 232, 283]))
            # self._get_tgt_permutation_idx(indices) : (tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]), tensor([3, 0, 1, 2, 1, 6, 7, 0, 3, 5, 2, 4]))
            indices, loss_weights, indices_rgb, indices_t, loss_class_weights, loss_bbox_weights, loss_giou_weights = self.matcher(outputs_without_aux, targets, outputs_rgb_without_aux, outputs_t_without_aux)
            indices_teacher = None

        if self.distill:
            loss_weights = None

        aux_indices_list = list()
        aux_indices_list_rgb = list()
        aux_indices_list_t = list()
        aux_indices_list_teacher = list()

        aux_loss_weights_list = list()

        aux_loss_class_weights_list = list()
        aux_loss_bbox_weights_list = list()
        aux_loss_giou_weights_list = list()

        aux_loss_weights_list_rgb = list()

        aux_loss_class_weights_list_rgb = list()
        aux_loss_bbox_weights_list_rgb = list()
        aux_loss_giou_weights_list_rgb = list()

        aux_loss_weights_list_t = list()

        aux_loss_class_weights_list_t = list()
        aux_loss_bbox_weights_list_t = list()
        aux_loss_giou_weights_list_t = list()

        if self.distill:
            if self.follow_teacher:
                for i, (aux_outputs, aux_outputs_rgb, aux_outputs_t) in enumerate(zip(outputs['aux_outputs'], outputs_rgb['aux_outputs'], outputs_t['aux_outputs'])):
                    aux_indices, aux_loss_weights, aux_indices_rgb, aux_indices_t, aux_loss_class_weights, aux_loss_bbox_weights, aux_loss_giou_weights = self.matcher(aux_outputs, targets, aux_outputs_rgb, aux_outputs_t)
                    aux_indices_list.append(aux_indices)
                aux_indices_list_teacher = aux_indices_list
            else:
                for i, aux_outputs in enumerate(outputs_distill['aux_outputs']):
                    aux_indices, aux_loss_weights, _, __,___,____,_____ = self.matcher(aux_outputs, targets, None, None)
                    aux_indices_list.append(aux_indices)
                for i, (aux_outputs, aux_outputs_rgb, aux_outputs_t) in enumerate(zip(outputs['aux_outputs'], outputs_rgb['aux_outputs'], outputs_t['aux_outputs'])):
                    aux_indices, aux_loss_weights, aux_indices_rgb, aux_indices_t, aux_loss_class_weights, aux_loss_bbox_weights, aux_loss_giou_weights = self.matcher(aux_outputs, targets, aux_outputs_rgb, aux_outputs_t)
                    aux_indices_list_teacher.append(aux_indices)          
        else:
            if outputs_rgb is not None and outputs_t is not None:
                for i, (aux_outputs, aux_outputs_rgb, aux_outputs_t) in enumerate(zip(outputs['aux_outputs'], outputs_rgb['aux_outputs'], outputs_t['aux_outputs'])):
                    aux_indices, aux_loss_weights, aux_indices_rgb, aux_indices_t, aux_loss_class_weights, aux_loss_bbox_weights, aux_loss_giou_weights = self.matcher(aux_outputs, targets, aux_outputs_rgb, aux_outputs_t)

                    aux_indices_list.append(aux_indices)

                    if aux_indices_rgb is not None:
                        aux_indices_list_rgb.append(aux_indices_rgb)

                    if aux_indices_t is not None:
                        aux_indices_list_t.append(aux_indices_t)

                    if aux_loss_weights is not None:
                        if self.instance_reweight:
                            aux_loss_weights_list.append(aux_loss_weights[0])
                            aux_loss_weights_list_rgb.append(aux_loss_weights[1])
                            aux_loss_weights_list_t.append(aux_loss_weights[2])
                            if self.term_reweight:
                                aux_loss_class_weights_list.append(aux_loss_class_weights[0])
                                aux_loss_class_weights_list_rgb.append(aux_loss_class_weights[1])
                                aux_loss_class_weights_list_t.append(aux_loss_class_weights[2])

                                aux_loss_bbox_weights_list.append(aux_loss_bbox_weights[0])
                                aux_loss_bbox_weights_list_rgb.append(aux_loss_bbox_weights[1])
                                aux_loss_bbox_weights_list_t.append(aux_loss_bbox_weights[2])

                                aux_loss_giou_weights_list.append(aux_loss_giou_weights[0])
                                aux_loss_giou_weights_list_rgb.append(aux_loss_giou_weights[1])
                                aux_loss_giou_weights_list_t.append(aux_loss_giou_weights[2])
                        else:
                            aux_loss_weights_list.append(aux_loss_weights[:, 0])  # aux_loss_weights[:, 0]形状为bs
                            aux_loss_weights_list_rgb.append(aux_loss_weights[:, 1])
                            aux_loss_weights_list_t.append(aux_loss_weights[:, 2])
            else:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    aux_indices, aux_loss_weights, _, __,___,____,_____ = self.matcher(aux_outputs, targets, None, None)
                    aux_indices_list.append(aux_indices)

        loss_weights_class_fusion = None
        loss_weights_class_rgb = None
        loss_weights_class_t = None

        loss_weights_bbox_fusion = None
        loss_weights_bbox_rgb = None 
        loss_weights_bbox_t = None 

        loss_weights_giou_fusion = None 
        loss_weights_giou_rgb = None 
        loss_weights_giou_t = None 

        if loss_weights is not None and dynamic_weight:
            loss_weights_fusion = loss_weights[0] if self.instance_reweight else loss_weights[:, 0]
            loss_weights_rgb = loss_weights[1] if self.instance_reweight else loss_weights[:, 1]
            loss_weights_t = loss_weights[2] if self.instance_reweight else loss_weights[:, 2]

            if self.term_reweight:
                loss_weights_class_fusion = loss_class_weights[0] 
                loss_weights_class_rgb = loss_class_weights[1] 
                loss_weights_class_t = loss_class_weights[2] 

                loss_weights_bbox_fusion = loss_bbox_weights[0] 
                loss_weights_bbox_rgb = loss_bbox_weights[1] 
                loss_weights_bbox_t = loss_bbox_weights[2] 

                loss_weights_giou_fusion = loss_giou_weights[0] 
                loss_weights_giou_rgb = loss_giou_weights[1] 
                loss_weights_giou_t = loss_giou_weights[2] 
        else:
            loss_weights_fusion = loss_weights_rgb = loss_weights_t = None
            aux_loss_weights_list = aux_loss_weights_list_rgb = aux_loss_weights_list_t = None

        if self.distill:
            loss = self.forward_single(outputs_distill, targets, indices, indices_teacher, None, None, None, None, aux_indices_list, aux_indices_list_teacher,
                                    None, None, None, None)
            return None, None, None, loss
        else:
            loss = self.forward_single(outputs, targets, indices, None, loss_weights_fusion, loss_weights_class_fusion, loss_weights_bbox_fusion, loss_weights_giou_fusion, aux_indices_list, None,
                                    aux_loss_weights_list, aux_loss_class_weights_list, aux_loss_bbox_weights_list, aux_loss_giou_weights_list, True)

        if self.prototype and rgb_prototypes != None:
            loss.update(self.get_prototype_loss(outputs, outputs_rgb, outputs_t, targets, indices, rgb_prototypes, t_prototypes, fusion_prototypes, aux_indices_list))

        # 计算红外和可见光分支的Loss
        loss_rgb = None
        loss_t = None
        if outputs_rgb is not None and outputs_t is not None:
            if indices_rgb is not None:
                loss_rgb = self.forward_single(outputs_rgb, targets, indices_rgb, None, loss_weights_rgb, loss_weights_class_rgb, loss_weights_bbox_rgb, loss_weights_giou_rgb, aux_indices_list_rgb, None,
                                               aux_loss_weights_list_rgb, aux_loss_class_weights_list_rgb, aux_loss_bbox_weights_list_rgb, aux_loss_giou_weights_list_rgb, False)
            else:
                loss_rgb = self.forward_single(outputs_rgb, targets, indices, None, loss_weights_rgb, loss_weights_class_rgb, loss_weights_bbox_rgb, loss_weights_giou_rgb, aux_indices_list, None,
                                               aux_loss_weights_list_rgb, aux_loss_class_weights_list_rgb, aux_loss_bbox_weights_list_rgb, aux_loss_giou_weights_list_rgb, False)
            if indices_t is not None:
                loss_t = self.forward_single(outputs_t, targets, indices_t, None, loss_weights_t, loss_weights_class_t, loss_weights_bbox_t, loss_weights_giou_t, aux_indices_list_t, None,
                                             aux_loss_weights_list_t, aux_loss_class_weights_list_t, aux_loss_bbox_weights_list_t, aux_loss_giou_weights_list_t, False)
            else:
                loss_t = self.forward_single(outputs_t, targets, indices, None, loss_weights_t, loss_weights_class_t, loss_weights_bbox_t, loss_weights_giou_t, aux_indices_list, None,
                                             aux_loss_weights_list_t, aux_loss_class_weights_list_t, aux_loss_bbox_weights_list_t, aux_loss_giou_weights_list_t, False)

        return loss, loss_rgb, loss_t, None
    
    def get_prototype_loss(self, outputs, outputs_rgb, outputs_t, targets, indices, rgb_prototypes, t_prototypes, fusion_prototypes, aux_indices_list):
        return_loss = dict()

        hs = outputs['hs']
        hs_rgb = outputs_rgb['hs_rgb']
        hs_t = outputs_t['hs_t']

        last_hs = hs[-1]
        last_hs_rgb = hs_rgb[-1]
        last_hs_t = hs_t[-1]

        loss_last_layer = self.get_prototype_loss_4_single_layer(indices, targets, last_hs, last_hs_rgb, last_hs_t, rgb_prototypes[-1], t_prototypes[-1], fusion_prototypes[-1])
        return_loss['loss_prototype'] = loss_last_layer

        if self.prototype_all_layers:
            layers_num = hs.shape[0]
            for i, (cur_hs, cur_hs_rgb, cur_hs_t) in enumerate(zip(hs, hs_rgb, hs_t)):
                if i == layers_num - 1:
                    break
                loss_cur_layer = self.get_prototype_loss_4_single_layer(aux_indices_list[i], targets, cur_hs, cur_hs_rgb, cur_hs_t, rgb_prototypes[i], t_prototypes[i], fusion_prototypes[i])
                return_loss['loss_prototype' + f'_{i}'] = loss_cur_layer
        
        return return_loss

    def get_prototype_loss_4_single_layer(self, indices, targets, cur_hs, cur_hs_rgb, cur_hs_t, rgb_prototypes, t_prototypes, fusion_prototypes):
        loss = torch.zeros(size=(1,), device=rgb_prototypes.device, dtype=rgb_prototypes.dtype)
        softmax = nn.Softmax(dim=1)

        for i, (indice, target, last_hs_i, last_hs_rgb_i, last_hs_t_i) in enumerate(zip(indices, targets, cur_hs, cur_hs_rgb, cur_hs_t)): # i索引batch
            positive_oq_feature = last_hs_i[indice[0]]  # n * 256, 其中n为真值框的数量
            positive_oq_feature_rgb = last_hs_rgb_i[indice[0]]
            positive_oq_feature_t = last_hs_t_i[indice[0]]

            labels = target[self.gt_field_class][indice[1]] # n,其中n为真值框的数量

            if not self.split_cls_reg:
                fusion_sim = -EU_dist(positive_oq_feature, fusion_prototypes) # n, num_classes, 其中n为真值框的数量, num_classes为
                rgb_sim = -EU_dist(positive_oq_feature_rgb, rgb_prototypes)
                t_sim = -EU_dist(positive_oq_feature_t, t_prototypes)
            else:
                hidden_dim = positive_oq_feature.shape[-1]
                fusion_sim = -EU_dist(positive_oq_feature[..., hidden_dim // 2:], fusion_prototypes) # n, num_classes, 其中n为真值框的数量, num_classes为
                rgb_sim = -EU_dist(positive_oq_feature_rgb[..., hidden_dim // 2:], rgb_prototypes)
                t_sim = -EU_dist(positive_oq_feature_t[..., hidden_dim // 2:], t_prototypes)

            score_fusion_p = [softmax(fusion_sim)[i][labels[i]] for i in range(fusion_sim.size(0))]  # 长度为n的列表，记录了融合分支中根据与泛型的欧式距离得到的置信度
            score_rgb_p = [softmax(rgb_sim)[i][labels[i]] for i in range(rgb_sim.size(0))]
            score_t_p = [softmax(t_sim)[i][labels[i]] for i in range(t_sim.size(0))]

            if not self.prototype_reweight:
                weights_fusion = [self.prototype_alpha] * labels.shape[0]
                weights_rgb = [self.prototype_alpha] * labels.shape[0]
                weights_t = [self.prototype_alpha] * labels.shape[0]
            else:
                weights_fusion = list()
                weights_rgb = list()
                weights_t = list()

                for score_fusion, score_rgb, score_t in zip(score_fusion_p, score_rgb_p, score_t_p):
                    max_score = max(score_fusion, score_rgb, score_t)
                    p_fusion = max_score / score_fusion
                    p_rgb = max_score / score_rgb
                    p_t = max_score / score_t

                    assert p_fusion == 1.0 or p_rgb == 1.0 or p_t == 1.0

                    weights_fusion.append(get_weight_from_p(p_fusion, self.prototype_alpha, self.adaptive_reweight))
                    weights_rgb.append(get_weight_from_p(p_rgb, self.prototype_alpha, self.adaptive_reweight))
                    weights_t.append(get_weight_from_p(p_t, self.prototype_alpha, self.adaptive_reweight))

                    # if score_fusion > score_rgb and score_fusion > score_t:
                    #     weights_fusion.append(0)
                    #     weights_rgb.append(self.prototype_alpha)
                    #     weights_t.append(self.prototype_alpha)
                    # elif score_rgb > score_fusion and score_rgb > score_t:
                    #     weights_fusion.append(self.prototype_alpha)
                    #     weights_rgb.append(0)
                    #     weights_t.append(self.prototype_alpha)
                    # elif score_t > score_fusion and score_t > score_rgb:
                    #     weights_fusion.append(self.prototype_alpha)
                    #     weights_rgb.append(self.prototype_alpha)
                    #     weights_t.append(0)
                    # else:
                    #     print("特殊情况出现了，有两个及以上分支的置信度相等，暂时将系数全部设置为prototype_alpha")
                    #     weights_fusion.append(self.prototype_alpha)
                    #     weights_rgb.append(self.prototype_alpha)
                    #     weights_t.append(self.prototype_alpha)

            weights_fusion = torch.as_tensor(weights_fusion, device=fusion_sim.device, dtype=fusion_sim.dtype)
            weights_rgb = torch.as_tensor(weights_rgb, device=rgb_sim.device, dtype=rgb_sim.dtype)
            weights_t = torch.as_tensor(weights_t, device=t_sim.device, dtype=t_sim.dtype)

            ceLoss = nn.CrossEntropyLoss(reduction='none')

            loss_fusion = ceLoss(fusion_sim, labels) * weights_fusion
            loss_rgb = ceLoss(rgb_sim, labels) * weights_rgb
            loss_t = ceLoss(t_sim, labels) * weights_t

            loss += torch.mean(loss_fusion)
            loss += torch.mean(loss_rgb)
            loss += torch.mean(loss_t)
        return torch.tensor(loss.item(), device=loss.device, dtype=loss.dtype)

def get_weight_from_p(p, alpha, adaptive_reweight=False):
    if p == 1.0:
        return 0.0
    else:
        if not adaptive_reweight:
            return alpha
        weight = 1.0 if p - 1 > 1.0 else p - 1
        return max(weight, 0.0)



