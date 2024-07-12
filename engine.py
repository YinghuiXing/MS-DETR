# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import sys
import torch
import shutil
import cv2

import os
import util.misc as utils
from util import box_ops

from typing import Iterable
from datasets.kaist import aggregate_detections
from pathlib import Path
from collections import defaultdict
from evaluation_script.evaluation_script import evaluate as evaluate2
from evaluation_script.evaluation_script import draw_all
from eval.kaist_eval_full import kaist_eval_full
import numpy as np
from thop import profile
import torch.distributed as dist


def save_tensor_to_file(tensor, file_path):
    """
    将PyTorch张量保存到文件中

    Args:
        tensor (torch.Tensor): 要保存的张量
        file_path (str): 保存文件的路径
    """
    with open(file_path, 'w') as file:
        file.write(str(tensor))

def find_integer_in_tensor(j, tensor):
    """
    检查整数 j 是否在张量 tensor 中，并返回其下标位置

    Args:
        j (int): 要检查的整数
        tensor (torch.Tensor): 输入张量

    Returns:
        (bool, int): 如果 j 存在于 tensor 中，则返回 True 和 j 在 tensor 中的下标位置，否则返回 False 和 None
    """
    # 使用 torch.eq 函数来检查整数是否在张量中
    result = torch.eq(tensor, j)
    indices = torch.nonzero(result, as_tuple=False)

    if indices.size(0) > 0:
        return True, indices.item()
    else:
        return False, None

def cal_prototype_4_single_layer(outputs, outputs_rgb, outputs_t, layer, decoder_layers, matcher, targets, gt_field_class, num_classes, rgb_prototypes_new, t_prototypes_new, fusion_prototypes_new, count_class, split_cls_reg):
    hidden_dim = rgb_prototypes_new.shape[-1]
    hs = outputs['hs'][layer]
    hs_rgb = outputs_rgb['hs_rgb'][layer]
    hs_t = outputs_t['hs_t'][layer]

    if layer == decoder_layers - 1:
        outputs_4_match = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        if outputs_rgb is not None and outputs_t is not None:
            outputs_rgb_4_match = {k: v for k, v in outputs_rgb.items() if k != 'aux_outputs' and k != 'enc_outputs'}
            outputs_t_4_match = {k: v for k, v in outputs_t.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        else:
            outputs_rgb_4_match = None
            outputs_t_4_match = None
    else:
        outputs_4_match = outputs['aux_outputs'][layer]
        outputs_rgb_4_match = outputs_rgb['aux_outputs'][layer]
        outputs_t_4_match = outputs_t['aux_outputs'][layer]
                
    # indices的样例：[(tensor([130, 271]), tensor([0, 1])), (tensor([  8,  18,  66,  68, 147]), tensor([3, 2, 0, 4, 1]))]
    indices, _, __, ___ = matcher(outputs_4_match, targets, outputs_rgb_4_match, outputs_t_4_match)
    for i, (indice, target, hs_i, hs_rgb_i, hs_t_i) in enumerate(zip(indices, targets, hs, hs_rgb, hs_t)):  # 此处的i索引batch
        cnt = 0
        assert_cnt = indice[0].shape[0]

        for j, (feature, feature_rgb, feature_t) in enumerate(zip(hs_i, hs_rgb_i, hs_t_i)):  # 此处的j索引object queries
            in_flag, ind = find_integer_in_tensor(j, indice[0])
            if in_flag:
                label = target[gt_field_class][indice[1][ind]]
                cnt += 1
            else:
                label = num_classes - 1
            if split_cls_reg:
                fusion_prototypes_new[layer, label, :] += feature[hidden_dim:]
                rgb_prototypes_new[layer, label, :] += feature_rgb[hidden_dim:]
                t_prototypes_new[layer, label, :] += feature_t[hidden_dim:]
            else:
                fusion_prototypes_new[layer, label, :] += feature
                rgb_prototypes_new[layer, label, :] += feature_rgb
                t_prototypes_new[layer, label, :] += feature_t
            count_class[layer][label] += 1

        assert cnt == assert_cnt

def list_eq(list_a, list_b):
    if len(list_a) != len(list_b):
        return False
    else:
        for a, b in zip(list_a, list_b):
            if a != b:
                return False
        return True

def cal_prototype(dataloader, model, matcher, device, epoch, cfg, rgb_prototypes_old, t_prototypes_old, fusion_prototypes_old):
    num_classes = cfg.MODEL.num_classes
    hidden_dim = cfg.MODEL.detr_hidden_dim
    decoder_layers = cfg.MODEL.TRANSFORMER.DECODER.layers 
    momentum_coef = cfg.MODEL.PROTOTYPE.momentum_coef
    begin_epoch = cfg.MODEL.PROTOTYPE.begin_epoch
    proportion = cfg.MODEL.PROTOTYPE.proportion
    split_cls_reg = cfg.MODEL.MS_DETR.split_cls_reg
    if epoch < begin_epoch:
        return None, None, None

    rgb_prototypes_new = torch.zeros(decoder_layers, num_classes, hidden_dim if not split_cls_reg else hidden_dim // 2).to(device)
    t_prototypes_new = torch.zeros(decoder_layers, num_classes, hidden_dim if not split_cls_reg else hidden_dim // 2).to(device)
    fusion_prototypes_new = torch.zeros(decoder_layers, num_classes, hidden_dim  if not split_cls_reg else hidden_dim // 2).to(device)

    count_class = [[0 for _ in range(num_classes)] for __ in range(decoder_layers)]

    # model.eval()
    model.train()
    with torch.no_grad():
        sample_count = 0
        all_num = len(dataloader)

        for *samples, targets in dataloader:
            samples = [(item.to(device) if item is not None else None) for item in samples]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=cfg.EXPERIMENT.amp):
                outputs, outputs_rgb, outputs_t = model(*samples, targets)
               
            for i in range(decoder_layers):
                cal_prototype_4_single_layer(outputs, outputs_rgb, outputs_t, i, decoder_layers, matcher, targets, cfg.MODEL.LOSS.gt_field_class, num_classes, rgb_prototypes_new, t_prototypes_new, fusion_prototypes_new, count_class, split_cls_reg)
                # print(count_class)
                if i != 0:
                    assert list_eq(count_class[i], count_class[0]), count_class
   
            sample_count += 1

            if sample_count >= all_num * proportion:
                break

        for c in range(num_classes):
            fusion_prototypes_new[:, c, :] /= count_class[0][c]
            rgb_prototypes_new[:, c, :] /= count_class[0][c]
            t_prototypes_new[:, c, :] /= count_class[0][c]
        
        # save_tensor_to_file(fusion_prototypes_new, "fusion_" + str(cfg.DISTRIBUTED.rank) + "_before.txt")
        dist.all_reduce(fusion_prototypes_new)
        dist.all_reduce(rgb_prototypes_new)
        dist.all_reduce(t_prototypes_new)
        # save_tensor_to_file(fusion_prototypes_new, "fusion_" + str(cfg.DISTRIBUTED.rank) + "_after.txt")

        world_size = cfg.DISTRIBUTED.world_size
        fusion_prototypes_new /= world_size
        rgb_prototypes_new /= world_size
        t_prototypes_new /= world_size

        if epoch != begin_epoch:
            fusion_prototypes_new = (1 - momentum_coef) * fusion_prototypes_new + momentum_coef * fusion_prototypes_old
            rgb_prototypes_new = (1 - momentum_coef) * rgb_prototypes_new + momentum_coef * rgb_prototypes_old
            t_prototypes_new = (1 - momentum_coef) * t_prototypes_new + momentum_coef * t_prototypes_old
        
        return rgb_prototypes_new, t_prototypes_new, fusion_prototypes_new

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, cfg=None, logger=None, ema_m=None,
                    rgb_prototypes=None, t_prototypes=None,fusion_prototypes=None, distill=False):
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.EXPERIMENT.amp)
    need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # temp = defaultdict(list)

    _cnt = 0
    dynamic_weight = False
    if (cfg.MODEL.MS_DETR.rgb_branch or cfg.MODEL.MS_DETR.t_branch) and epoch >= cfg.MODEL.LOSS.start_dynamic_weight:
        dynamic_weight = True

    for *samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        samples = [(item.to(device) if item is not None else None) for item in samples]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=cfg.EXPERIMENT.amp):
            if need_tgt_for_training:
                outputs, outputs_rgb, outputs_t, outputs_distill = model(*samples, targets)
            else:
                outputs, outputs_rgb, outputs_t, outputs_distill = model(*samples, None)

            # loss_dict, loss_dict_rgb, loss_dict_t = criterion(outputs, outputs_rgb, outputs_t, targets, dynamic_weight)
            loss_dict, loss_dict_rgb, loss_dict_t, loss_dict_distill = criterion(outputs, outputs_rgb, outputs_t, outputs_distill, targets, dynamic_weight, rgb_prototypes, t_prototypes, fusion_prototypes)
            weight_dict = criterion.weight_dict
            if distill:
                losses = sum(loss_dict_distill[k] * weight_dict[k] for k in loss_dict_distill.keys() if k in weight_dict)
            else:
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                if loss_dict_rgb is not None and loss_dict_t is not None:
                    losses_rgb = sum(loss_dict_rgb[k] * weight_dict[k] for k in loss_dict_rgb.keys() if k in weight_dict)
                    losses_t = sum(loss_dict_t[k] * weight_dict[k] for k in loss_dict_t.keys() if k in weight_dict)
                    losses = losses_rgb + losses + losses_t

        # reduce losses over all GPUs for logging purposes
        loss_value = 0
        loss_rgb_value = 0
        loss_t_value = 0
        loss_distill_value = 0

        if distill:
            loss_dict_distill_reduced = utils.reduce_dict(loss_dict_distill)
            loss_dict_distill_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_distill_reduced.items()}
            loss_dict_distill_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_distill_reduced.items() if k in weight_dict}
            losses_distill_reduced_scaled = sum(loss_dict_distill_reduced_scaled.values())

            loss_distill_value = losses_distill_reduced_scaled.item()
        else:
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

        if loss_dict_rgb is not None and loss_dict_t is not None:
            loss_dict_rgb_reduced = utils.reduce_dict(loss_dict_rgb)
            loss_dict_rgb_reduced_unscaled = {f'{k}_unscaled_rgb': v for k, v in loss_dict_rgb_reduced.items()}
            loss_dict_rgb_reduced_scaled = {f'{k}_rgb': v * weight_dict[k] for k, v in loss_dict_rgb_reduced.items() if k in weight_dict}
            losses_rgb_reduced_scaled = sum(loss_dict_rgb_reduced_scaled.values())
            loss_rgb_value = losses_rgb_reduced_scaled.item()

            loss_dict_t_reduced = utils.reduce_dict(loss_dict_t)
            loss_dict_t_reduced_unscaled = {f'{k}_unscaled_t': v for k, v in loss_dict_t_reduced.items()}
            loss_dict_t_reduced_scaled = {f'{k}_t': v * weight_dict[k] for k, v in loss_dict_t_reduced.items() if k in weight_dict}
            losses_t_reduced_scaled = sum(loss_dict_t_reduced_scaled.values())
            loss_t_value = losses_t_reduced_scaled.item()

        if distill and not math.isfinite(loss_distill_value):
            print("Loss is {}, stopping training".format(loss_distill_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if not math.isfinite(loss_value) or not math.isfinite(loss_rgb_value) or not math.isfinite(loss_t_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if cfg.EXPERIMENT.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if distill:
            metric_logger.update(loss=loss_distill_value, **loss_dict_distill_reduced_scaled, **loss_dict_distill_reduced_unscaled)
            if 'class_error' in loss_dict_distill_reduced:
                metric_logger.update(class_error=loss_dict_distill_reduced['class_error'])
        else:
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            if 'class_error' in loss_dict_reduced:
                metric_logger.update(class_error=loss_dict_reduced['class_error'])
            if loss_dict_rgb is not None and loss_dict_t is not None:
                metric_logger.update(loss_rgb=loss_rgb_value, loss_t=loss_t_value,
                                    **loss_dict_rgb_reduced_scaled, **loss_dict_rgb_reduced_unscaled,
                                    **loss_dict_t_reduced_scaled, **loss_dict_t_reduced_unscaled,)
                if 'class_error' in loss_dict_rgb_reduced and 'class_error' in loss_dict_t_reduced:
                    metric_logger.update(class_error_rgb=loss_dict_rgb_reduced['class_error'])
                    metric_logger.update(class_error_t=loss_dict_t_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, postprocessors, data_loader, device, flops , cfg=None, distill=False, info_weights=False):
    model.eval()
    res = dict()
    res_rgb = dict()
    res_t = dict()
    res_distill = dict()

    # mAP相关指标
    jdict, stats, ap, ap_class = [], [], [], []
    jdict_rgb, stats_rgb, ap_rgb, ap_class_rgb = [], [], [], []
    jdict_t, stats_t, ap_t, ap_class_t = [], [], [], []

    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()


    for *samples, targets in data_loader:
        """
            samples: is a tuple, like (NestedTensor, NestedTensor), NestedTensor.tensors.shape == [batch_size, 3, H, W]
            targets: is a tuple, like (Dict, Dict, ..., Dict) and its length == batch_size
        """

        samples = [(item.to(device) if item is not None else None) for item in samples]

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)  # shape: [batch_size, 2]
        orig_target_sizes = orig_target_sizes.to(device)
        flip_directions = [t['flip_direction'] for t in targets]

        with torch.cuda.amp.autocast(enabled=cfg.EXPERIMENT.amp):
            if flops:
                flops, params = profile(model, inputs=(*samples, targets))
                print(f"FLOPs: {flops / 1e9} GFlops")  # 打印计算量（以十亿次浮点运算为单位）  
                print(f"Params: {params / 1e6} M")  # 打印参数量（以百万为单位）
                # print(f"FLOPs: {flops} ")  # 打印计算量（以十亿次浮点运算为单位）  
                # print(f"Params: {params} M")
                print("=" * 50)
                # from mmengine.analysis import get_model_complexity_info
                # analysis_results = get_model_complexity_info(model, inputs=(*samples,))
                # print(analysis_results)
                exit(0)  # yangshuo 2024.4.13

            outputs, outputs_rgb, outputs_t, outputs_distill = model(*samples, targets)

            # from visualize_draw_point import draw_point 
            # # 可视化
            # pred_logits = outputs['pred_logits'] 
            # pred_boxes = outputs['pred_boxes']
            # points = outputs['points_list'][-1]  # 我们只要最后一层decoder的输出
            # weights = outputs['weights_list'][-1] # 我们只要最后一层decoder的输出
            # img_rgb_path_list  = [d['img_absolute_path_rgb'] for d in targets]
            # img_t_path_list = [d['img_absolute_path_t'] for d in targets]
            # draw_point(pred_logits , points , weights , img_rgb_path_list , img_t_path_list )
            # i += 1 
            # if i == 300 : break
            # continue
            

            if cfg.MODEL.MS_DETR.SEGMENTATION.flag and cfg.MODEL.MS_DETR.SEGMENTATION.vis:
                visualize_seg(targets, outputs, cfg.EXPERIMENT.output_dir)
        """
            results is a list that contains batch_size dicts whose keys are scores, labels and boxes
            res is a dict whose keys are file_name and values are corresponding det results like {scores:[num_selects], labels:[num_selects], boxes:[num_selects, 4]}
        """

        if distill:
            results_distill = postprocessors['bbox'](outputs_distill, orig_target_sizes, flip_directions)

            if cfg.TEST.metric == 'ap':
                calculateApStat4Batch(results, targets, iouv, niou, stats, cfg)
            else:
                res_distill.update({target['anno_absolute_path']: output for target, output in zip(targets, results_distill)})
        else:
            if outputs is not None:
                results = postprocessors['bbox'](outputs, orig_target_sizes, flip_directions, info_weights)

                if cfg.TEST.metric == 'ap':
                    calculateApStat4Batch(results, targets, iouv, niou, stats, cfg)
                else:
                    res.update({target['anno_absolute_path']: output for target, output in zip(targets, results)})

            if cfg.MODEL.MS_DETR.rgb_branch and outputs_rgb is not None:
                results_rgb = postprocessors['bbox'](outputs_rgb, orig_target_sizes, flip_directions)

                if cfg.TEST.metric == 'ap':
                    calculateApStat4Batch(results_rgb, targets, iouv, niou, stats_rgb, cfg)
                else:
                    res_rgb.update({target['anno_absolute_path']: output for target, output in zip(targets, results_rgb)})

            if cfg.MODEL.MS_DETR.t_branch and outputs_t is not None:
                results_t = postprocessors['bbox'](outputs_t, orig_target_sizes, flip_directions)

                if cfg.TEST.metric == 'ap':
                    calculateApStat4Batch(results_t, targets, iouv, niou, stats_t, cfg)
                else:
                    res_t.update({target['anno_absolute_path']: output for target, output in zip(targets, results_t)})
    if cfg.TEST.metric == 'ap':
        return stats, stats_rgb, stats_t , None
    else:
        return res, res_rgb, res_t, res_distill


def visualize_seg(targets, outputs, output_path):
    from util.misc import interpolate
    from visualize.vis_tools import drawBBoxes

    if len(outputs['pred_masks_rgb']):
        for i, masks in enumerate(outputs['pred_masks_rgb']):
            for mask, target in zip(masks, targets):
                mask = mask.sigmoid()
                image_ind = target['image_ind']
                mask_saved_path = os.path.join(output_path, 'visualize_masks', 'rgb', str(image_ind) + '_' + str(i) + '.jpg')
                mask = interpolate(mask[None], size=[512, 640], mode="nearest")[0, 0].to('cpu')
                bboxes = target['gt_bboxes_rgb'].to('cpu')
                drawBBoxes(mask, bboxes, bboxes_mode='xywh', saved_path=mask_saved_path, percentile=True)

    if len(outputs['pred_masks_t']):
        for i, masks in enumerate(outputs['pred_masks_t']):
            for mask, target in zip(masks, targets):
                mask = mask.sigmoid()
                image_ind = target['image_ind']
                mask_saved_path = os.path.join(output_path, 'visualize_masks', 't', str(image_ind) + '_' + str(i) + '.jpg')
                mask = interpolate(mask[None], size=[512, 640], mode="nearest")[0, 0].to('cpu')
                bboxes = target['gt_bboxes_rgb'].to('cpu')
                drawBBoxes(mask, bboxes, bboxes_mode='xywh', saved_path=mask_saved_path, percentile=True)

    if len(outputs['pred_masks_fusion']):
        for i, masks in enumerate(outputs['pred_masks_fusion']):
            for mask, target in zip(masks, targets):
                mask = mask.sigmoid()
                image_ind = target['image_ind']
                mask_saved_path = os.path.join(output_path, 'visualize_masks', 'fusion', str(image_ind) + '_' + str(i) + '.jpg')
                mask = interpolate(mask[None], size=[512, 640], mode="nearest")[0, 0].to('cpu')
                bboxes = target['gt_bboxes_rgb'].to('cpu')
                drawBBoxes(mask, bboxes, bboxes_mode='xywh', saved_path=mask_saved_path, percentile=True)


def ap_eval(stats, nc=1):
    dt, p, r, f1, mp, mr, map50, map75, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False)
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 1 + '%11.3g' * 5  # print format
    print(pf % ('all', nt.sum(), mp, mr, map50, map75, map))


def ap_per_class(tp, conf, pred_cls, target_cls, plot = False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    # names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    # names = {i: v for i, v in enumerate(names)}  # to dict

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def calculateApStat4Batch(results, targets, iouv, niou, stats, cfg):
    for i, (result, target) in enumerate(zip(results, targets)):
        nl = target[cfg.MODEL.LOSS.gt_field_bbox].shape[0]  # 真值总数
        tcls = target[cfg.MODEL.LOSS.gt_field_class].tolist() if nl else []  # 真值的类标签，类型为list

        orig_size = target['orig_size']

        img_h, img_w = orig_size.unbind(0)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])

        predn = torch.cat((result['boxes'].cpu(), result['scores'][:, None].cpu(), result['labels'][:, None].cpu()), dim=1)  # dt, [N, 6]
        if nl:
            tbox = box_ops.box_cxcywh_to_xyxy(target[cfg.MODEL.LOSS.gt_field_bbox] * scale_fct[None, :])
            labelsn = torch.cat((target[cfg.MODEL.LOSS.gt_field_class][:, None], tbox), 1)  # gt, [M, 5]
            correct = process_batch(predn, labelsn, iouv)
        else:
            correct = torch.zeros(predn.shape[0], niou, dtype=torch.bool)

        stats.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), tcls))


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


@torch.no_grad()
def inference(model, dataset, device, args=None, ind=0):
    samples_rgb, samples_t, targets = dataset[ind]
    samples_rgb = samples_rgb.unsqueeze(0)
    samples_t = samples_t.unsqueeze(0)
    samples_rgb = samples_rgb.to(device)
    samples_t = samples_t.to(device)

    orig_target_sizes = targets['orig_size'].unsqueeze(0)  # shape: [batch_size, 2]
    orig_target_sizes = orig_target_sizes.to(device)

    with torch.cuda.amp.autocast(enabled=args.amp):
        outputs, outputs_rgb, outputs_t = model(samples_rgb, samples_t)

    points_list = outputs['points_list']
    weights_list = outputs['weights_list']
    init_reference = outputs['init_reference']
    inter_references = outputs['inter_references']

    H_ = orig_target_sizes[0, 0].item()
    W_ = orig_target_sizes[0, 1].item()

    results, topk_boxes = post_process(outputs, orig_target_sizes)
    scores, labels, boxes = results[0]['scores'], results[0]['labels'], results[0]['boxes']

    base_name = os.path.basename(targets['annotation_id']).split('.')[0]
    [set_id, video_id, image_id] = base_name.split('_')
    img_path_t = os.path.join(args.kaist_root, 'Images', set_id, video_id, 'lwir', image_id + '.jpg')
    img_path_rgb = os.path.join(args.kaist_root, 'Images', set_id, video_id, 'visible', image_id + '.jpg')
    output_dir = os.path.join(args.output_dir, 'inference', base_name)

    pedestrian_boxes = list()
    pedestrian_scores = list()
    object_queries_ids = list()

    for i, (score, label, box) in enumerate(zip(scores, labels, boxes)):
        score = score.item()
        label = label.item()
        if label != 0:
            continue
        if score < 0.5:
            continue
        x0, y0, x1, y1 = box[0].item(), box[1].item(), box[2].item(), box[3].item()

        if y1 - y0 < 80:
            continue
        pedestrian_boxes.append([x0, y0, x1, y1])
        pedestrian_scores.append(score)
        object_queries_ids.append(topk_boxes[0][i].item())

    if len(pedestrian_boxes) == 0:
        print('这张图像没有检测出行人!')
        return

    for i in range(args.dec_layers):
        points = points_list[i]  # [bs, num_queries, n_heads, n_levels, n_points, 2]
        weights = weights_list[i]  # [bs, num_queries, n_heads, n_levels, n_points]

        if i == 0:
            anchor = box_ops.box_cxcywh_to_xyxy(init_reference) * torch.as_tensor([W_, H_, W_, H_], device=init_reference.device)
        else:
            anchor = (box_ops.box_cxcywh_to_xyxy(inter_references[i - 1]) * torch.as_tensor([W_, H_, W_, H_], device=inter_references.device))[0]  # [bs, num_queries, 4]

        n_heads = points.shape[2]
        n_levels = points.shape[3]
        for j, (pede_box, pede_score, oq_ind) in enumerate(zip(pedestrian_boxes, pedestrian_scores, object_queries_ids)):
            for k in range(n_heads):
                cur_weight = weights[0, oq_ind, k, :, :]
                cur_weight_rgb = cur_weight[:3, :].view(-1)
                cur_weight_t = cur_weight[3:, :].view(-1)
                key_points_rgb = points[0, oq_ind, k, :3, :, :].view(-1, 2)
                key_points_t = points[0, oq_ind, k, 3:, :, :].view(-1, 2)

                # max_v = torch.max(cur_weight)
                # min_v = torch.min(cur_weight)
                # cur_weight = (cur_weight - min_v) / (max_v - min_v)
                cur_weight_rgb_top_2, top_2_ind_rgb = torch.topk(cur_weight_rgb, 2)
                cur_weight_t_top_2, top_2_ind_t = torch.topk(cur_weight_t, 2)

                key_points_rgb_top2 = (key_points_rgb[top_2_ind_rgb] * torch.as_tensor([W_, H_], device=points.device)).tolist()
                key_points_t_top2 = (key_points_t[top_2_ind_t] * torch.as_tensor([W_, H_], device=points.device)).tolist()

                cur_anchor = anchor[oq_ind, :].tolist()
                saved_path_rgb = os.path.join(output_dir, 'dec_' + str(i), 'det_' + str(j), 'head_' + str(k), 'rgb.jpg')
                saved_path_t = os.path.join(output_dir, 'dec_' + str(i), 'det_' + str(j), 'head_' + str(k), 't.jpg')
                draw_boxes_and_points(cur_anchor, key_points_rgb_top2, cur_weight_rgb_top_2.tolist(), pede_box, img_path_rgb, saved_path_rgb)
                draw_boxes_and_points(cur_anchor, key_points_t_top2, cur_weight_t_top_2.tolist(), pede_box, img_path_t,
                                      saved_path_t)


def post_process(outputs, target_sizes):
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

    return results, topk_boxes


def draw_boxes_and_points(anchor, key_points, key_point_weights, pede_box, img_path, saved_path):
    img = cv2.imread(img_path)
    cv2.rectangle(img, (int(anchor[0]), int(anchor[1])), (int(anchor[2]), int(anchor[3])), (0, 0, 255), 1)
    #cv2.rectangle(img, (int(pede_box[0]), int(pede_box[1])), (int(pede_box[2]), int(pede_box[3])), (255, 255, 0), 1)

    for point, weight in zip(key_points, key_point_weights):
        r_, g_, b_, _ = plt.get_cmap("seismic", 100)(int(weight * 100))

        r_value = int(r_ * 255)
        g_value = int(g_ * 255)
        b_value = int(b_ * 255)

        color = (b_value, g_value, r_value)
        cv2.circle(img, (int(point[0]), int(point[1])), 1, color, 4)

    if saved_path is not None:
        saved_dir = os.path.dirname(saved_path)
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)
        cv2.imwrite(saved_path, img)
    else:
        cv2.imshow(os.path.basename(img_path), img)
        cv2.waitKey()
        cv2.destroyAllWindows()


def kaist_eval(res, output_dir, resume, branch_name, info_attention=False):
    output_dir = os.path.join(output_dir, 'test')
    checkpoint = os.path.basename(resume).split('.')[0]
    dataset_type = 'test'
    # set_Names = ['Reasonable', 'All', 'Far', 'Medium', 'Near', 'None', 'Partial', 'Heavy']
    set_Names = ['Reasonable', 'All']

    output_dir_05, output_dir = writeDetResult(res, output_dir, checkpoint, branch_name, False, info_attention) # output_dir :/data/wangsong/results/23_9_26/exp47/test/checkpoint/fusion_branch/tenth/det
    savedPaths, isEmpty = aggregate_detections(output_dir, dataset_type=dataset_type, cvc14=False) # savedPaths: "test-all": "/data/wangsong/results/23_9_26/exp47/test/checkpoint/fusion_branch/tenth/det-test-all.txt", ...
    all_path = list()
    all_path.append(savedPaths['test-all'])

    results = [evaluate2('KAIST_annotation.json', rstFile) for rstFile in all_path]
    return_results = [_[1] for _ in results]
    results = [_[0] for _ in results]
    for ind, set_name in enumerate(set_Names):
        tmp_results = [r[set_name] for r in results]
        tmp_results = sorted(tmp_results, key=lambda x: x['all'].summarize(ind), reverse=True)
        results_img_path = os.path.join(output_dir, '..', branch_name + '_result_' + set_name + '.pdf')
        draw_all(tmp_results, filename=results_img_path)

    return return_results


def cvc14_eval(res, output_dir, resume, branch_name):
    output_dir = os.path.join(output_dir, 'test')
    checkpoint = os.path.basename(resume).split('.')[0]

    output_dir_05, output_dir = writeDetResult(res, output_dir, checkpoint, branch_name, True)
    # result = kaist_eval_full(output_dir, '/home/Newdisk/zhangshizhou/datasets/CVC-14/gt/CVC14_annotations', 'test', True)
    result = kaist_eval_full(output_dir, '/data/wangsong/datasets/CVC-14/gt/CVC14_annotations', 'test', True)
    return result


def writeDetResult(res, output_dir, checkpoint, branch_name, CVC=False, info_attention=False):
    output_dir = os.path.join(output_dir, checkpoint)
    output_dir = os.path.join(output_dir, branch_name)
    output_dir_1 = os.path.join(output_dir, 'half', 'det')
    output_dir_2 = os.path.join(output_dir, 'tenth', 'det')

    Path(output_dir_1).mkdir(parents=True, exist_ok=True)
    Path(output_dir_2).mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        if CVC:
            day_or_night = os.path.dirname(k).split('/')[-4]
            if day_or_night == 'Day':
                day_or_night = 'day_'
            elif day_or_night == 'Night':
                day_or_night = 'night_'
            base_name = day_or_night + os.path.basename(k).split('.')[0] + '.txt'
        else:
            base_name = os.path.basename(k).split('.')[0] + '.txt'

        f1 = open(os.path.join(output_dir_1, base_name), 'w')
        f2 = open(os.path.join(output_dir_2, base_name), 'w')

        lines_1 = list()
        lines_2 = list()

        if info_attention:
            for score, label, box, attention_rgb, attention_t in zip(v['scores'], v['labels'], v['boxes'], v['attention_rgb'], v['attention_t']):
                score = score.item()
                label = label.item()
                attention_rgb = attention_rgb.item()
                attention_t = attention_t.item()
                if label != 0:
                    continue
                label = 'person'
                x0, y0, x1, y1 = box[0].item(), box[1].item(), box[2].item(), box[3].item()

                line = ' '.join((str(label), str(x0), str(y0), str(x1), str(y1), str(score), str(attention_rgb), str(attention_t),'\n'))
                lines_2.append(line)
                if score >= 0.5:
                    lines_1.append(line)
        else:
            for score, label, box in zip(v['scores'], v['labels'], v['boxes']):
                score = score.item()
                label = label.item()
                if label != 0:
                    continue
                label = 'person'
                x0, y0, x1, y1 = box[0].item(), box[1].item(), box[2].item(), box[3].item()

                line = ' '.join((str(label), str(x0), str(y0), str(x1), str(y1), str(score), '\n'))
                lines_2.append(line)
                if score >= 0.5:
                    lines_1.append(line)
        f1.writelines(lines_1)
        f2.writelines(lines_2)
        f1.close()
        f2.close()
    return output_dir_1, output_dir_2


def visualizeDetResultThreeBranch(res, res_rgb, res_t, output_dir, checkpoint, kaist_root):
    output_dir = os.path.join(output_dir, 'det_visualize_three_branch')
    output_dir = os.path.join(output_dir, checkpoint)

    for k in res.keys():
        value = res[k]
        value_rgb = res_rgb[k]
        value_t = res_t[k]

        [set_id, video_id, image_id] = (os.path.basename(k).split('.')[0]).split('_')
        rgb_image_path = os.path.join(kaist_root, 'Images', set_id, video_id, 'visible', image_id + '.jpg')
        t_image_path = os.path.join(kaist_root, 'Images', set_id, video_id, 'lwir', image_id + '.jpg')

        i = 0
        for score, label, box, score_rgb, label_rgb, box_rgb, score_t, label_t, box_t in zip(value['scores'], value['labels'], value['boxes'], value_rgb['scores'], value_rgb['labels'], value_rgb['boxes'], value_t['scores'], value_t['labels'], value_t['boxes']):
            score = round(score.item(), 3)
            score_rgb = round(score_rgb.item(), 3)
            score_t = round(score_t.item(), 3)

            if score < 0.1 and score_rgb < 0.1 and score_t < 0.1:
                continue
            residual = abs(score_rgb - score_t)
            if residual < 0.05:
                continue
            print('yes')
            name = set_id + '_' + video_id + '_' + image_id + '_' + str(i) + '.jpg'
            saved_path_rgb = os.path.join(output_dir, 'rgb', name)
            saved_path_t = os.path.join(output_dir, 't', name)

            x0, y0, x1, y1 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            box = [x0, y0, x1 - x0, y1 - y0]
            x0_rgb, y0_rgb, x1_rgb, y1_rgb = box_rgb[0].item(), box_rgb[1].item(), box_rgb[2].item(), box_rgb[3].item()
            box_rgb = [x0_rgb, y0_rgb, x1_rgb - x0_rgb, y1_rgb - y0_rgb]
            x0_t, y0_t, x1_t, y1_t = box_t[0].item(), box_t[1].item(), box_t[2].item(), box_t[3].item()
            box_t = [x0_t, y0_t, x1_t - x0_t, y1_t - y0_t]

            box_paint = [box, 'pink', 1.0, '']
            box_paint_rgb = [box_rgb, 'green', 1.0, '']
            box_paint_t = [box_t, 'blue', 1.0, str(round(residual, 2)) + ',' + str(round(score_rgb, 2)) + ',' + str(round(score_t, 2))]
            paintBBoxes(rgb_image_path, [box_paint, box_paint_rgb, box_paint_t], saved_path_rgb)
            paintBBoxes(t_image_path, [box_paint, box_paint_rgb, box_paint_t], saved_path_t)
            i += 1


def visualizeDetResult(res, output_dir, checkpoint, threshold, gt_dir, kaist_root, boxes_field_name='boxes', branch_name='fusion', paint=True, only_paint_correct=False):
    output_dir = os.path.join(output_dir, 'det_visualize', checkpoint, 'threshold' + '_' + str(threshold), branch_name)

    total_det = 0
    error_det_far = 0
    error_det_middle = 0
    error_det_near = 0

    ignore_det_far = 0
    ignore_det_middle = 0
    ignore_det_near = 0

    det_far = defaultdict(int)
    det_middle = defaultdict(int)
    det_near = defaultdict(int)

    total_gt = 0
    ignore_gt = 0
    miss_gt_far = 0
    miss_gt_middle = 0
    miss_gt_near = 0

    for k, v in res.items():
        base_name = os.path.basename(k).split('.')[0] + '.txt'

        [set_id, video_id, image_id] = (os.path.basename(k).split('.')[0]).split('_')
        rgb_image_path = os.path.join(kaist_root, 'Images', set_id, video_id, 'visible', image_id + '.jpg')
        t_image_path = os.path.join(kaist_root, 'Images', set_id, video_id, 'lwir', image_id + '.jpg')

        name = set_id + '_' + video_id + '_' + image_id + '.jpg'

        dts = list()

        for score, label, box in zip(v['scores'], v['labels'], v[boxes_field_name]):
            score = round(score.item(), 2)
            label = label.item()
            if label != 0:
                continue
            if score < threshold:
                continue
            x0, y0, x1, y1 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            dt = [x0, y0, x1 - x0, y1 - y0, score]
            dts.append(dt)

        gt_file_path = os.path.join(gt_dir, 'test-all', 'annotations', base_name)

        with open(gt_file_path) as f:
            gts_str = f.readlines()

        del gts_str[0]
        gts = list()

        for gt_str in gts_str:
            gt_data = gt_str.strip().split()
            gt_label = gt_data[0]
            x0_gt, y0_gt, w_gt, h_gt = float(gt_data[1]), float(gt_data[2]), float(gt_data[3]), float(gt_data[4])
            occlusion = int(gt_data[5])

            ignore = 0

            if gt_label != 'person':
                ignore = 1
            elif h_gt < 55:
                ignore = 1
            elif occlusion == 2:
                ignore = 1
            elif x0_gt < 5 or x0_gt > 635 or (x0_gt + w_gt) < 5 or (x0_gt + w_gt) > 635:
                ignore = 1
            elif y0_gt < 5 or y0_gt > 507 or (y0_gt + h_gt) < 5 or (y0_gt + h_gt) > 507:
                ignore = 1

            gts.append([x0_gt, y0_gt, w_gt, h_gt, ignore])

        if len(dts) > 0:
            dts = np.array(dts)
        else:
            dts = None

        if len(gts) > 0:
            gts = np.array(gts)
        else:
            gts = None

        gts, dts = evalRes(gts, dts)

        total_det += len(dts)
        total_gt += len(gts)

        cur_total_det = list()
        cur_correct_det = list()
        cur_error_det_far = list()
        cur_error_det_middle = list()
        cur_error_det_near = list()

        cur_ignore_det_far = list()
        cur_ignore_det_middle = list()
        cur_ignore_det_near = list()

        cur_det_far = defaultdict(list)
        cur_det_middle = defaultdict(list)
        cur_det_near = defaultdict(list)

        cur_miss_gt_far = list()
        cur_miss_gt_middle = list()
        cur_miss_gt_near = list()

        for dt in dts:
            box = list(dt[:4])
            height = box[3]
            score = dt[4]
            match = dt[5]

            if match == -1:
                color = 'green'
            elif match == 0:
                color = 'red'
            elif match == 1:
                color = 'green'

            box_paint = [box, color, 1.0, str(round(score, 2))]
            cur_total_det.append(box_paint)

            if match == -1:
                color = 'pink'
                box_paint = [box, color, 1.0, str(score)]
                if 0 < height < 55:
                    cur_ignore_det_far.append(box_paint)
                    ignore_det_far += 1
                elif 55 <= height < 115:
                    cur_ignore_det_middle.append(box_paint)
                    ignore_det_middle += 1
                elif height >= 115:
                    cur_ignore_det_near.append(box_paint)
                    ignore_det_near += 1
            elif match == 0:
                color = 'red'
                box_paint = [box, color, 1.0, str(score)]

                if 0 < height < 55:
                    cur_error_det_far.append(box_paint)
                    error_det_far += 1
                elif 55 <= height < 115:
                    cur_error_det_middle.append(box_paint)
                    error_det_middle += 1
                elif height >= 115:
                    cur_error_det_near.append(box_paint)
                    error_det_near += 1
            elif match == 1:
                color = 'green'
                box_paint = [box, color, 1.0, str(score)]

                def score_split(cur_score):
                    if 0.5 <= cur_score < 0.55:
                        return '50'
                    elif 0.55 <= cur_score < 0.6:
                        return '55'
                    elif 0.6 <= cur_score < 0.65:
                        return '60'
                    elif 0.65 <= cur_score < 0.7:
                        return '65'
                    elif 0.7 <= cur_score < 0.75:
                        return '70'
                    elif 0.75 <= cur_score < 0.8:
                        return '75'
                    elif 0.8 <= cur_score < 0.85:
                        return '80'
                    elif 0.85 <= cur_score < 0.9:
                        return '85'
                    elif 0.9 <= cur_score < 0.95:
                        return '90'
                    elif 0.95 <= cur_score <= 1:
                        return '95'

                split_set = score_split(score)
                if 0 < height < 55:
                    cur_det_far[split_set].append(box_paint)
                    det_far[split_set] += 1
                elif 55 <= height < 115:
                    cur_det_middle[split_set].append(box_paint)
                    det_middle[split_set] += 1
                elif height >= 115:
                    cur_det_near[split_set].append(box_paint)
                    det_near[split_set] += 1

        for gt in gts:
            box = list(gt[:4])
            height = box[3]
            match = gt[4]

            if match == -1:
                ignore_gt += 1
            elif match == 0:
                color = 'red'
                box_paint = [box, color, 1.0, '']

                if 0 < height < 55:
                    cur_miss_gt_far.append(box_paint)
                    miss_gt_far += 1
                elif 55 <= height < 115:
                    cur_miss_gt_middle.append(box_paint)
                    miss_gt_middle += 1
                elif height >= 115:
                    cur_miss_gt_near.append(box_paint)
                    miss_gt_near += 1
        if len(cur_error_det_far) == 0 and len(cur_error_det_middle) == 0 and len(cur_error_det_near) == 0 and len(cur_miss_gt_far) == 0 and len(cur_miss_gt_middle) == 0 and len(cur_miss_gt_near) == 0:
            cur_correct_det += cur_total_det

        if paint:
            if only_paint_correct:
                if len(cur_correct_det) > 0:
                    print('找到一张完全检测正确的图像')
                    paintBBoxes(rgb_image_path, cur_correct_det, os.path.join(output_dir, 'correct', 'visible', name))
                    paintBBoxes(t_image_path, cur_correct_det, os.path.join(output_dir, 'correct', 'lwir', name))
            else:
                if len(cur_total_det) > 0:
                    paintBBoxes(rgb_image_path, cur_total_det, os.path.join(output_dir, 'total', 'visible', name))
                    paintBBoxes(t_image_path, cur_total_det, os.path.join(output_dir, 'total', 'lwir', name))

                if len(cur_error_det_far) > 0:
                    paintBBoxes(rgb_image_path, cur_error_det_far, os.path.join(output_dir, 'error', 'far', 'visible',name))
                    paintBBoxes(t_image_path, cur_error_det_far, os.path.join(output_dir, 'error', 'far', 'lwir', name))

                if len(cur_error_det_middle) > 0:
                    paintBBoxes(rgb_image_path, cur_error_det_middle, os.path.join(output_dir, 'error', 'middle', 'visible',name))
                    paintBBoxes(t_image_path, cur_error_det_middle, os.path.join(output_dir, 'error', 'middle', 'lwir', name))

                if len(cur_error_det_near) > 0:
                    paintBBoxes(rgb_image_path, cur_error_det_near, os.path.join(output_dir, 'error', 'near', 'visible',name))
                    paintBBoxes(t_image_path, cur_error_det_near, os.path.join(output_dir, 'error', 'near', 'lwir', name))

                if len(cur_ignore_det_far) > 0:
                    paintBBoxes(rgb_image_path, cur_ignore_det_far, os.path.join(output_dir, 'ignore', 'far', 'visible',name))
                    paintBBoxes(t_image_path, cur_ignore_det_far, os.path.join(output_dir, 'ignore', 'far', 'lwir', name))

                if len(cur_ignore_det_middle) > 0:
                    paintBBoxes(rgb_image_path, cur_ignore_det_middle, os.path.join(output_dir, 'ignore', 'middle', 'visible',name))
                    paintBBoxes(t_image_path, cur_ignore_det_middle, os.path.join(output_dir, 'ignore', 'middle', 'lwir', name))

                if len(cur_ignore_det_near) > 0:
                    paintBBoxes(rgb_image_path, cur_ignore_det_near, os.path.join(output_dir, 'ignore', 'near', 'visible',name))
                    paintBBoxes(t_image_path, cur_ignore_det_near, os.path.join(output_dir, 'ignore', 'near', 'lwir', name))

                for k, v in cur_det_far.items():
                    if len(v) > 0:
                        paintBBoxes(rgb_image_path, v, os.path.join(output_dir, 'score_' + k, 'far', 'visible',name))
                        paintBBoxes(t_image_path, v, os.path.join(output_dir, 'score_' + k, 'far', 'lwir', name))

                for k, v in cur_det_middle.items():
                    if len(v) > 0:
                        paintBBoxes(rgb_image_path, v, os.path.join(output_dir, 'score_' + k, 'middle', 'visible',name))
                        paintBBoxes(t_image_path, v, os.path.join(output_dir, 'score_' + k, 'middle', 'lwir', name))

                for k, v in cur_det_near.items():
                    if len(v) > 0:
                        paintBBoxes(rgb_image_path, v, os.path.join(output_dir, 'score_' + k, 'near', 'visible',name))
                        paintBBoxes(t_image_path, v, os.path.join(output_dir, 'score_' + k, 'near', 'lwir', name))

                if len(cur_miss_gt_far) > 0:
                    paintBBoxes(rgb_image_path, cur_miss_gt_far, os.path.join(output_dir, 'miss', 'far', 'visible',name))
                    paintBBoxes(t_image_path, cur_miss_gt_far, os.path.join(output_dir, 'miss', 'far', 'lwir', name))

                if len(cur_miss_gt_middle) > 0:
                    paintBBoxes(rgb_image_path, cur_miss_gt_middle, os.path.join(output_dir, 'miss', 'middle', 'visible',name))
                    paintBBoxes(t_image_path, cur_miss_gt_middle, os.path.join(output_dir, 'miss', 'middle', 'lwir', name))

                if len(cur_miss_gt_near) > 0:
                    paintBBoxes(rgb_image_path, cur_miss_gt_near, os.path.join(output_dir, 'miss', 'near', 'visible',name))
                    paintBBoxes(t_image_path, cur_miss_gt_near, os.path.join(output_dir, 'miss', 'near', 'lwir', name))

    print('total_det:', total_det)
    print('error_det_far:', error_det_far)
    print('error_det_middle:', error_det_middle)
    print('error_det_near:', error_det_near)

    print('ignore_det_far:', ignore_det_far)
    print('ignore_det_middle:', ignore_det_middle)
    print('ignore_det_near:', ignore_det_near)

    print('det_far:')
    print(det_far)
    print('det_middle:')
    print(det_middle)
    print('det_near:')
    print(det_near)

    print('total_gt:', total_gt)
    print('ignore_gt:', ignore_gt)
    print('miss_gt_far:', miss_gt_far)
    print('miss_gt_middle:', miss_gt_middle)
    print('miss_gt_near:', miss_gt_near)


def visualizeDetResultCVC14(res, output_dir, checkpoint, threshold, gt_dir, cvc14_root, boxes_field_name='boxes', paint=True, only_paint_correct=True):
    output_dir = os.path.join(output_dir, 'det_visualize')
    output_dir = os.path.join(output_dir, boxes_field_name)
    output_dir = os.path.join(output_dir, checkpoint)

    total_det = 0
    error_det_far = 0
    error_det_middle = 0
    error_det_near = 0

    ignore_det_far = 0
    ignore_det_middle = 0
    ignore_det_near = 0

    det_far = defaultdict(int)
    det_middle = defaultdict(int)
    det_near = defaultdict(int)

    total_gt = 0
    ignore_gt = 0
    miss_gt_far = 0
    miss_gt_middle = 0
    miss_gt_near = 0

    for k, v in res.items():
        [day_or_night, _, train_or_test, __, anno_name] = k.split('/')
        img_name = anno_name.replace('.txt', '.tif')

        rgb_image_path = os.path.join(cvc14_root, day_or_night, 'Visible', train_or_test, 'FramesPos', img_name)
        t_image_path = os.path.join(cvc14_root, day_or_night, 'FIR', train_or_test, 'FramesPos', img_name)

        dts = list()

        for score, label, box in zip(v['scores'], v['labels'], v[boxes_field_name]):
            score = round(score.item(), 2)
            label = label.item()
            if label != 0:
                continue
            if score < 0.1:
                continue
            x0, y0, x1, y1 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            dt = [x0, y0, x1 - x0, y1 - y0, score]
            dts.append(dt)

        gt_name = 'day_' + img_name.replace('.tif', '.txt') if day_or_night == 'Day' else 'night_' + img_name.replace('.tif', '.txt')
        gt_file_path = os.path.join(gt_dir, 'test-all', 'annotations', gt_name)

        with open(gt_file_path) as f:
            gts_str = f.readlines()

        del gts_str[0]
        gts = list()

        for gt_str in gts_str:
            gt_data = gt_str.strip().split()
            gt_label = gt_data[0]
            x0_gt, y0_gt, w_gt, h_gt = float(gt_data[1]), float(gt_data[2]), float(gt_data[3]), float(gt_data[4])
            occlusion = int(gt_data[5])

            ignore = 0

            if gt_label != 'person':
                ignore = 1
            elif h_gt < 55:
                ignore = 1
            elif occlusion == 2:
                ignore = 1
            elif x0_gt < 5 or x0_gt > 635 or (x0_gt + w_gt) < 5 or (x0_gt + w_gt) > 635:
                ignore = 1
            elif y0_gt < 5 or y0_gt > 466 or (y0_gt + h_gt) < 5 or (y0_gt + h_gt) > 466:
                ignore = 1

            gts.append([x0_gt, y0_gt, w_gt, h_gt, ignore])

        if len(dts) > 0:
            dts = np.array(dts)
        else:
            dts = None

        if len(gts) > 0:
            gts = np.array(gts)
        else:
            gts = None

        gts, dts = evalRes(gts, dts)

        total_det += len(dts)
        total_gt += len(gts)

        cur_total_det = list()
        cur_error_det_far = list()
        cur_error_det_middle = list()
        cur_error_det_near = list()

        cur_ignore_det_far = list()
        cur_ignore_det_middle = list()
        cur_ignore_det_near = list()

        cur_det_far = defaultdict(list)
        cur_det_middle = defaultdict(list)
        cur_det_near = defaultdict(list)

        cur_miss_gt_far = list()
        cur_miss_gt_middle = list()
        cur_miss_gt_near = list()

        for dt in dts:
            box = list(dt[:4])
            height = box[3]
            score = dt[4]
            match = dt[5]

            if match == -1:
                color = 'pink'
            elif match == 0:
                color = 'red'
            elif match == 1:
                color = 'green'

            box_paint = [box, color, 1.0, str(round(score, 2))]
            cur_total_det.append(box_paint)

            if match == -1:
                color = 'pink'
                box_paint = [box, color, 1.0, str(score)]
                if 0 < height < 55:
                    cur_ignore_det_far.append(box_paint)
                    ignore_det_far += 1
                elif 55 <= height < 115:
                    cur_ignore_det_middle.append(box_paint)
                    ignore_det_middle += 1
                elif height >= 115:
                    cur_ignore_det_near.append(box_paint)
                    ignore_det_near += 1
            elif match == 0:
                color = 'red'
                box_paint = [box, color, 1.0, str(score)]

                if 0 < height < 55:
                    cur_error_det_far.append(box_paint)
                    error_det_far += 1
                elif 55 <= height < 115:
                    cur_error_det_middle.append(box_paint)
                    error_det_middle += 1
                elif height >= 115:
                    cur_error_det_near.append(box_paint)
                    error_det_near += 1
            elif match == 1:
                color = 'green'
                box_paint = [box, color, 1.0, str(score)]

                def score_split(cur_score):
                    if 0.5 <= cur_score < 0.55:
                        return '50'
                    elif 0.55 <= cur_score < 0.6:
                        return '55'
                    elif 0.6 <= cur_score < 0.65:
                        return '60'
                    elif 0.65 <= cur_score < 0.7:
                        return '65'
                    elif 0.7 <= cur_score < 0.75:
                        return '70'
                    elif 0.75 <= cur_score < 0.8:
                        return '75'
                    elif 0.8 <= cur_score < 0.85:
                        return '80'
                    elif 0.85 <= cur_score < 0.9:
                        return '85'
                    elif 0.9 <= cur_score < 0.95:
                        return '90'
                    elif 0.95 <= cur_score <= 1:
                        return '95'
                    else:
                        return '45'

                split_set = score_split(score)
                if 0 < height < 55:
                    cur_det_far[split_set].append(box_paint)
                    det_far[split_set] += 1
                elif 55 <= height < 115:
                    cur_det_middle[split_set].append(box_paint)
                    det_middle[split_set] += 1
                elif height >= 115:
                    cur_det_near[split_set].append(box_paint)
                    det_near[split_set] += 1

        for gt in gts:
            box = list(gt[:4])
            height = box[3]
            match = gt[4]

            if match == -1:
                ignore_gt += 1
            elif match == 0:
                color = 'red'
                box_paint = [box, color, 1.0, '']

                if 0 < height < 55:
                    cur_miss_gt_far.append(box_paint)
                    miss_gt_far += 1
                elif 55 <= height < 115:
                    cur_miss_gt_middle.append(box_paint)
                    miss_gt_middle += 1
                elif height >= 115:
                    cur_miss_gt_near.append(box_paint)
                    miss_gt_near += 1

        # if paint:
        #     if only_paint_correct:
        #         if len(cur_error_det_far) == 0 and len(cur_error_det_middle) == 0 and len(cur_error_det_near) == 0 and len(cur_ignore_det_far) == 0 and len(cur_ignore_det_middle) == 0 and len(cur_ignore_det_near) == 0:
        #             if len(cur_total_det) > 0:
        #                 paintBBoxes(rgb_image_path, cur_total_det,
        #                             os.path.join(output_dir, 'correct', 'visible', img_name))
        #                 paintBBoxes(t_image_path, cur_total_det, os.path.join(output_dir, 'correct', 'lwir', img_name))
        #     else:
        #         if len(cur_total_det) > 0:
        #             paintBBoxes(rgb_image_path, cur_total_det, os.path.join(output_dir, 'total', 'visible', img_name))
        #             paintBBoxes(t_image_path, cur_total_det, os.path.join(output_dir, 'total', 'lwir', img_name))
        #
        #         if len(cur_error_det_far) > 0:
        #             paintBBoxes(rgb_image_path, cur_error_det_far, os.path.join(output_dir, 'error', 'far', 'visible',img_name))
        #             paintBBoxes(t_image_path, cur_error_det_far, os.path.join(output_dir, 'error', 'far', 'lwir', img_name))
        #
        #         if len(cur_error_det_middle) > 0:
        #             paintBBoxes(rgb_image_path, cur_error_det_middle, os.path.join(output_dir, 'error', 'middle', 'visible',img_name))
        #             paintBBoxes(t_image_path, cur_error_det_middle, os.path.join(output_dir, 'error', 'middle', 'lwir', img_name))
        #
        #         if len(cur_error_det_near) > 0:
        #             paintBBoxes(rgb_image_path, cur_error_det_near, os.path.join(output_dir, 'error', 'near', 'visible',img_name))
        #             paintBBoxes(t_image_path, cur_error_det_near, os.path.join(output_dir, 'error', 'near', 'lwir', img_name))
        #
        #         if len(cur_ignore_det_far) > 0:
        #             paintBBoxes(rgb_image_path, cur_ignore_det_far, os.path.join(output_dir, 'ignore', 'far', 'visible',img_name))
        #             paintBBoxes(t_image_path, cur_ignore_det_far, os.path.join(output_dir, 'ignore', 'far', 'lwir', img_name))
        #
        #         if len(cur_ignore_det_middle) > 0:
        #             paintBBoxes(rgb_image_path, cur_ignore_det_middle, os.path.join(output_dir, 'ignore', 'middle', 'visible',img_name))
        #             paintBBoxes(t_image_path, cur_ignore_det_middle, os.path.join(output_dir, 'ignore', 'middle', 'lwir', img_name))
        #
        #         if len(cur_ignore_det_near) > 0:
        #             paintBBoxes(rgb_image_path, cur_ignore_det_near, os.path.join(output_dir, 'ignore', 'near', 'visible',img_name))
        #             paintBBoxes(t_image_path, cur_ignore_det_near, os.path.join(output_dir, 'ignore', 'near', 'lwir', img_name))
        #
        #         for k, v in cur_det_far.items():
        #             if len(v) > 0:
        #                 paintBBoxes(rgb_image_path, v, os.path.join(output_dir, 'score_' + k, 'far', 'visible',img_name))
        #                 paintBBoxes(t_image_path, v, os.path.join(output_dir, 'score_' + k, 'far', 'lwir', img_name))
        #
        #         for k, v in cur_det_middle.items():
        #             if len(v) > 0:
        #                 paintBBoxes(rgb_image_path, v, os.path.join(output_dir, 'score_' + k, 'middle', 'visible',img_name))
        #                 paintBBoxes(t_image_path, v, os.path.join(output_dir, 'score_' + k, 'middle', 'lwir', img_name))
        #
        #         for k, v in cur_det_near.items():
        #             if len(v) > 0:
        #                 paintBBoxes(rgb_image_path, v, os.path.join(output_dir, 'score_' + k, 'near', 'visible',img_name))
        #                 paintBBoxes(t_image_path, v, os.path.join(output_dir, 'score_' + k, 'near', 'lwir', img_name))
        #
        #         if len(cur_miss_gt_far) > 0:
        #             paintBBoxes(rgb_image_path, cur_miss_gt_far, os.path.join(output_dir, 'miss', 'far', 'visible',img_name))
        #             paintBBoxes(t_image_path, cur_miss_gt_far, os.path.join(output_dir, 'miss', 'far', 'lwir', img_name))
        #
        #         if len(cur_miss_gt_middle) > 0:
        #             paintBBoxes(rgb_image_path, cur_miss_gt_middle, os.path.join(output_dir, 'miss', 'middle', 'visible',img_name))
        #             paintBBoxes(t_image_path, cur_miss_gt_middle, os.path.join(output_dir, 'miss', 'middle', 'lwir', img_name))
        #
        #         if len(cur_miss_gt_near) > 0:
        #             paintBBoxes(rgb_image_path, cur_miss_gt_near, os.path.join(output_dir, 'miss', 'near', 'visible',img_name))
        #             paintBBoxes(t_image_path, cur_miss_gt_near, os.path.join(output_dir, 'miss', 'near', 'lwir', img_name))

    print('total_det:', total_det)
    print('error_det_far:', error_det_far)
    print('error_det_middle:', error_det_middle)
    print('error_det_near:', error_det_near)

    print('ignore_det_far:', ignore_det_far)
    print('ignore_det_middle:', ignore_det_middle)
    print('ignore_det_near:', ignore_det_near)

    print('det_far:')
    print(det_far)
    print('det_middle:')
    print(det_middle)
    print('det_near:')
    print(det_near)

    print('total_gt:', total_gt)
    print('ignore_gt:', ignore_gt)
    print('miss_gt_far:', miss_gt_far)
    print('miss_gt_middle:', miss_gt_middle)
    print('miss_gt_near:', miss_gt_near)


def visualizeFarPede(ours, sota_list, gt_dir, kaist_root, output_dir):
    def readDt(dt_file):
        dtBBoxes = list()
        detData = np.loadtxt(dt_file, delimiter=',')

        for num in range(2252):
            idx = detData[:, 0] == (num + 1)
            dtBBoxes.append(detData[idx][:, 1:])

        return dtBBoxes

    model_name = 'MS-DETR'
    dts = readDt(ours)  # List, len == 2252

    other_dts = dict()
    for sota in sota_list:
        cur_model_name = os.path.basename(sota).split('_')[0]
        cur_dts = readDt(sota)
        other_dts[cur_model_name] = cur_dts

    gtLoadConstraints = {
        'labels': ['person', ],
        'otherLabels': ['people', 'person?', 'cyclist'],
        'hRng': [1, float("inf")],
        'xRng': [5, 635],
        'yRng': [5, 507],
        'vType': ['none', 'partial', 'heavy']
    }

    gtBBoxes = list()  # List, len == 2252
    gtFilePaths, gtFileNames = getFiles(gt_dir)
    for path in gtFilePaths:
        gtBBoxes.append(load_gt_bbox(path, gtLoadConstraints, False)[1])

    for i in range(2252):
        cur_gts = gtBBoxes[i]
        cur_dts_ours = dts[i]
        cur_dts_mlpd = other_dts['MLPD'][i]
        cur_gts_ours, cur_dts_ours = evalRes(cur_gts, cur_dts_ours)
        cur_gts_mlpd, cur_dts_mlpd = evalRes(cur_gts, cur_dts_mlpd)

        continue_flag = True
        for gt_ours, gt_mlpd in zip(cur_gts_ours, cur_gts_mlpd):
            h = gt_ours[3]
            assert h == gt_mlpd[3]
            if h < 45:
                if gt_ours[4] and not gt_mlpd[4]:
                    continue_flag = False
                    break

        if continue_flag:
            continue
        print('yes')

        gtFileName = gtFileNames[i]
        [set_id, video_id, img_id] = gtFileName.split('.')[0].split('_')
        name = gtFileName.split('.')[0]
        img_path_rgb = os.path.join(kaist_root, 'Images', set_id, video_id, 'visible', img_id+'.jpg')
        img_path_t = os.path.join(kaist_root, 'Images', set_id, video_id, 'lwir', img_id + '.jpg')

        paint_boxes_list = list()
        paint_far_boxes_list = list()

        for gt in cur_gts:
            box = list(gt[:4])
            h = gt[3]
            box_paint = [box, 'yellow', 1.0, '']
            paint_boxes_list.append(box_paint)
            if h < 45:
                paint_far_boxes_list.append(box_paint)

        if len(paint_boxes_list) > 0:
            paintBBoxes(img_path_rgb, paint_boxes_list, os.path.join(output_dir, name, 'all', 'annotation-rgb.jpg'))
            paintBBoxes(img_path_t, paint_boxes_list, os.path.join(output_dir, name, 'all', 'annotation-t.jpg'))

        if len(paint_far_boxes_list) > 0:
            paintBBoxes(img_path_rgb, paint_far_boxes_list, os.path.join(output_dir, name, 'far', 'annotation-rgb.jpg'))
            paintBBoxes(img_path_t, paint_far_boxes_list, os.path.join(output_dir, name, 'far', 'annotation-t.jpg'))

        paint_boxes_list.clear()
        paint_far_boxes_list.clear()

        for dt in cur_dts_ours:
            box = list(dt[:4])
            score = round(dt[4], 1)
            h = dt[3]
            # box_paint = [box, 'green', 1.0, str(score)]
            box_paint = [box, 'green', 1.0, '']
            paint_boxes_list.append(box_paint)
            if h < 45:
                paint_far_boxes_list.append(box_paint)

        for gt in cur_gts_ours:
            box = list(gt[:4])
            h = gt[3]
            match = gt[4]
            if match != 1:
                box_paint = [box, 'red', 1.0, '']
                paint_boxes_list.append(box_paint)
                if h < 45:
                    paint_far_boxes_list.append(box_paint)

        if len(paint_boxes_list) > 0:
            paintBBoxes(img_path_rgb, paint_boxes_list, os.path.join(output_dir, name, 'all', 'MS-DETR-rgb.jpg'))
            paintBBoxes(img_path_t, paint_boxes_list, os.path.join(output_dir, name, 'all', 'MS-DETR-t.jpg'))

        if len(paint_far_boxes_list) > 0:
            paintBBoxes(img_path_rgb, paint_far_boxes_list, os.path.join(output_dir, name, 'far', 'MS-DETR-rgb.jpg'))
            paintBBoxes(img_path_t, paint_far_boxes_list, os.path.join(output_dir, name, 'far', 'MS-DETR-t.jpg'))

        for k, v in other_dts.items():
            paint_boxes_list.clear()
            paint_far_boxes_list.clear()

            cur_gts_certain, cur_dts_certain = evalRes(cur_gts, v[i])
            if k == 'MLPD':
                color = 'green'
            elif k == 'MBNet':
                color = 'green'
            elif k == 'ARCNN':
                color = 'green'
            for prediction in cur_dts_certain:
                box = list(prediction[:4])
                score = round(prediction[4], 1)
                match = prediction[5]
                if k in ('CIAN', 'ARCNN'):
                    score = round(score / 100, 1)
                h = prediction[3]
                if match == 1:
                    # box_paint = [box, 'green', 1.0, str(score)]
                    box_paint = [box, color, 1.0, '']
                    paint_boxes_list.append(box_paint)
                    if h < 46:
                        paint_far_boxes_list.append(box_paint)

            for gt in cur_gts_certain:
                box = list(gt[:4])
                h = gt[3]
                match = gt[4]
                if match != 1:
                    box_paint = [box, 'red', 1.0, '']
                    paint_boxes_list.append(box_paint)
                    if h < 45:
                        paint_far_boxes_list.append(box_paint)

            if len(paint_boxes_list) > 0:
                paintBBoxes(img_path_rgb, paint_boxes_list, os.path.join(output_dir, name, 'all', k+'-rgb.jpg'))
                paintBBoxes(img_path_t, paint_boxes_list, os.path.join(output_dir, name, 'all', k+'-t.jpg'))

            if len(paint_far_boxes_list) > 0:
                paintBBoxes(img_path_rgb, paint_far_boxes_list,
                            os.path.join(output_dir, name, 'far', k+'-rgb.jpg'))
                paintBBoxes(img_path_t, paint_far_boxes_list, os.path.join(output_dir, name, 'far', k+'-t.jpg'))












