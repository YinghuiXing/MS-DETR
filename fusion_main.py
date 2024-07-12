# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json

import random
import time
from pathlib import Path
import os, sys
# os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

from util.logger import setup_logger

import numpy as np
import torch

import util.misc as utils
from datasets import build_dataLoader
from engine import evaluate, train_one_epoch, kaist_eval, inference, visualizeDetResult, visualizeDetResultCVC14, ap_eval, cvc14_eval, cal_prototype
from models import build_model
from util.utils import clean_state_dict
from default import get_cfg_defaults
from collections import OrderedDict


def get_args_parser():
    parser = argparse.ArgumentParser('DAB-DETR', add_help=False)

    parser.add_argument('--exp_config', default='', type=str)
    parser.add_argument('--exp_config_student', default='', type=str)
    parser.add_argument('--visible_devices', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--find_unused_params', default=False, action='store_true')
    parser.add_argument('--action', default='train', type=str)
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')   # from --local_rank -> --local-rank   yangshuo 2024.4.2   BTW 这个参数其实不用指定
    parser.add_argument("--flops", action='store_true', default=False)
    parser.add_argument("--only_fusion", action='store_true', default=False)

    return parser

def main(output_dir, resume, find_unused_params, flops, only_fusion, cfg):
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    utils.init_distributed_mode(cfg.DISTRIBUTED)

    # setup logger
    logger = setup_logger(output=os.path.join(output_dir, 'info.txt'),
                          distributed_rank=cfg.DISTRIBUTED.rank, color=False, name="MS-DETR")

    # log config information
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: " + ' '.join(sys.argv))
    if cfg.DISTRIBUTED.rank == 0:
        save_yaml_path = os.path.join(output_dir, "config.yaml")
        with open(save_yaml_path, 'w') as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(save_yaml_path))

    device = torch.device(cfg.MODEL.device)

    # fix the seed for reproducibility
    seed = cfg.EXPERIMENT.seed + utils.get_rank()
    # seed = cfg.EXPERIMENT.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors= build_model(only_fusion, cfg.MODEL, cfg.DISTILL, cfg.REC)
    wo_class_error = False
    # print(device)
    model.to(device)

    # for name, parameter in model.named_parameters():
    #     print('ws', name, parameter.requires_grad)

    model_without_ddp = model
    if cfg.DISTRIBUTED.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.DISTRIBUTED.gpu],
                                                          find_unused_parameters=find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:' + str(n_parameters))
    logger.info(
        "params:\n" + json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.TRAIN.lr_backbone,
        }
    ]

    # optimizer = torch.optim.SGD(param_dicts, lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.weight_decay)
    # optimizer = torch.optim.SGD(param_dicts, lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.weight_decay, momentum=0.9)
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.lr_drop, gamma=cfg.TRAIN.lr_drop_rate)

    dataset, dataSampler, dataBatchSampler, dataLoader = build_dataLoader(cfg)

    output_dir = Path(output_dir)
    if resume:
        if resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(resume, map_location='cpu')
        _load_output = model_without_ddp.load_state_dict(checkpoint['model'])
        logger.info(str(_load_output))
        if cfg.EXPERIMENT.action != 'test' and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            cfg.EXPERIMENT.start_epoch = checkpoint['epoch'] + 1
    
    if not resume and cfg.REC.flag:
        assert cfg.REC.checkpoint
        checkpoint = torch.load(cfg.REC.checkpoint, map_location='cpu')['model']
        _load_output = model_without_ddp.load_state_dict(checkpoint, strict=False)
        print(_load_output)

    if not resume and cfg.DISTILL.flag:
        assert cfg.DISTILL.teacher_checkpoint
        checkpoint = torch.load(cfg.DISTILL.teacher_checkpoint, map_location='cpu')['model']
        distill_modality_rgb = cfg.DISTILL.distill_modality_rgb
        rec_flag = cfg.DISTILL.rec_fusion or cfg.DISTILL.rec_another
        keyMaps = getKeyMap4Distill(checkpoint, distill_modality_rgb, rec_flag)

        loadMaps = {k:checkpoint[v] for k,v in keyMaps.items()}

        for k in loadMaps.keys():
            if 'cross_attn.sampling_offsets.weight' in k and 'distill' in k:
                v = loadMaps[k]
                v = v.view(8, 8, 4, 2, 256)
                if distill_modality_rgb:
                    v = v[:, :4, ...].contiguous().view(256, 256)
                else:
                    v = v[:, 4:, ...].contiguous().view(256, 256)
                loadMaps[k] = v
            elif 'cross_attn.sampling_offsets.bias' in k and 'distill' in k:
                v = loadMaps[k]
                v = v.view(8, 8, 4, 2)
                if distill_modality_rgb:
                    v = v[:, :4, ...].contiguous().view(256)
                else:
                    v = v[:, 4:, ...].contiguous().view(256)
                loadMaps[k] = v
            elif 'cross_attn.attention_weights.weight' in k and 'distill' in k:
                v = loadMaps[k]
                v = v.view(8, 8, 4, 1, 256)
                if distill_modality_rgb:
                    v = v[:, :4, ...].contiguous().view(128, 256)
                else:
                    v = v[:, 4:, ...].contiguous().view(128, 256)
                loadMaps[k] = v
            elif 'cross_attn.attention_weights.bias' in k and 'distill' in k:
                v = loadMaps[k]
                v = v.view(8, 8, 4, 1)
                if distill_modality_rgb:
                    v = v[:, :4, ...].contiguous().view(128)
                else:
                    v = v[:, 4:, ...].contiguous().view(128)
                loadMaps[k] = v
        loadMaps = OrderedDict(loadMaps)
        _load_output = model_without_ddp.load_state_dict(loadMaps, strict=False)
        print(_load_output)

    if not resume and not cfg.DISTILL.flag and cfg.EXPERIMENT.pretrain_model_path:
        checkpoint = torch.load(cfg.EXPERIMENT.pretrain_model_path, map_location='cpu')['model']
        split_cls_reg = cfg.MODEL.MS_DETR.split_cls_reg
        _ignorekeywordlist = cfg.EXPERIMENT.finetune_ignore if cfg.EXPERIMENT.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))

        if cfg.EXPERIMENT.fit_flag:
            _tmp_st = OrderedDict(
                {k: v for k, v in clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
            _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        else:
            loadMaps = {k:checkpoint[v] for k,v in getKeyMap(checkpoint, split_cls_reg).items()}
            extraMaps = {}

            if cfg.MODEL.TRANSFORMER.DECODER.fusion:
                for k in loadMaps.keys():
                    if 'cross_attn.sampling_offsets.weight' in k:
                        v = loadMaps[k]
                        extraMaps[k.replace('cross_attn.sampling_offsets.weight', 'cross_attn.sampling_offsets_rgb.weight')] = v
                        extraMaps[k.replace('cross_attn.sampling_offsets.weight', 'cross_attn.sampling_offsets_t.weight')] = v
                        v = v.view(8, 4, 4, 2, 256)
                        if cfg.MODEL.TRANSFORMER.fusion_backbone_encoder:
                            v = torch.cat([v, v, v, v], dim=1).view(1024, 256)
                        else:
                            v = torch.cat([v, v], dim=1).view(512, 256)
                        loadMaps[k] = v
                    elif 'cross_attn.sampling_offsets.bias' in k:
                        v = loadMaps[k]
                        extraMaps[k.replace('cross_attn.sampling_offsets.bias', 'cross_attn.sampling_offsets_rgb.bias')] = v
                        extraMaps[k.replace('cross_attn.sampling_offsets.bias', 'cross_attn.sampling_offsets_t.bias')] = v
                        v = v.view(8, 4, 4, 2)
                        if cfg.MODEL.TRANSFORMER.fusion_backbone_encoder:
                            v = torch.cat([v, v, v, v], dim=1).view(1024)
                        else:
                            v = torch.cat([v, v], dim=1).view(512)
                        loadMaps[k] = v
                    elif 'cross_attn.attention_weights.weight' in k:
                        v = loadMaps[k]
                        extraMaps[k.replace('cross_attn.attention_weights.weight', 'cross_attn.attention_weights_rgb.weight')] = v
                        extraMaps[k.replace('cross_attn.attention_weights.weight', 'cross_attn.attention_weights_t.weight')] = v
                        v = v.view(8, 4, 4, 1, 256)
                        if cfg.MODEL.TRANSFORMER.fusion_backbone_encoder:
                            v = torch.cat([v, v, v, v], dim=1).view(512, 256)
                        else:
                            v = torch.cat([v, v], dim=1).view(256, 256)
                        loadMaps[k] = v
                    elif 'cross_attn.attention_weights.bias' in k:
                        v = loadMaps[k]
                        extraMaps[k.replace('cross_attn.attention_weights.bias', 'cross_attn.attention_weights_rgb.bias')] = v
                        extraMaps[k.replace('cross_attn.attention_weights.bias', 'cross_attn.attention_weights_t.bias')] = v
                        v = v.view(8, 4, 4, 1)
                        if cfg.MODEL.TRANSFORMER.fusion_backbone_encoder:
                            v = torch.cat([v, v, v, v], dim=1).view(512)
                        else:
                            v = torch.cat([v, v], dim=1).view(256)
                        loadMaps[k] = v
                    elif 'cross_attn.value_proj' in k:
                        v = loadMaps[k]
                        extraMaps[
                            k.replace('cross_attn.value_proj', 'cross_attn.value_proj_rgb')] = v
                        extraMaps[
                            k.replace('cross_attn.value_proj', 'cross_attn.value_proj_t')] = v
                    # elif 'query_embed.weight' in k :   # yangshuo 2024.4.8  消融query number实验  50
                    #     v = loadMaps[k] 
                    #     v = v[:50,:]
                    #     loadMaps[k] = v 

            if cfg.MODEL.TRANSFORMER.ENCODER.layers_4_fusion != 0:
                for k in loadMaps.keys():
                    if 'self_attn.sampling_offsets.weight' in k:
                        v = loadMaps[k]
                        v = v.view(8, 4, 4, 2, 256)
                        v = torch.cat([v, v], dim=1).view(512, 256)
                        loadMaps[k] = v
                    elif 'self_attn.sampling_offsets.bias' in k:
                        v = loadMaps[k]
                        v = v.view(8, 4, 4, 2)
                        v = torch.cat([v, v], dim=1).view(512)
                        loadMaps[k] = v
                    elif 'self_attn.attention_weights.weight' in k:
                        v = loadMaps[k]
                        v = v.view(8, 4, 4, 1, 256)
                        v = torch.cat([v, v], dim=1).view(256, 256)
                        loadMaps[k] = v
                    elif 'self_attn.attention_weights.bias' in k:
                        v = loadMaps[k]
                        v = v.view(8, 4, 4, 1)
                        v = torch.cat([v, v], dim=1).view(256)
                        loadMaps[k] = v

            loadMaps.update(extraMaps)
            loadMaps = OrderedDict(loadMaps)
            _load_output = model_without_ddp.load_state_dict(loadMaps, strict=False)
        logger.info(str(_load_output))
        # raise RuntimeError
        # import ipdb; ipdb.set_trace()

    if not resume and cfg.EXPERIMENT.pretrain_model_path_rgb and cfg.EXPERIMENT.pretrain_model_path_t:
        checkpoint_rgb = torch.load(cfg.EXPERIMENT.pretrain_model_path_rgb, map_location='cpu')['model']
        checkpoint_t = torch.load(cfg.EXPERIMENT.pretrain_model_path_t, map_location='cpu')['model']

        _ignorekeywordlist = cfg.EXPERIMENT.finetune_ignore if cfg.EXPERIMENT.finetune_ignore else []
        ignorelist = []
        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        _tmp_st_rgb = OrderedDict(
            {k: v for k, v in clean_state_dict(checkpoint_rgb).items() if check_keep(k, _ignorekeywordlist)})
        _tmp_st_t = OrderedDict(
            {k: v for k, v in clean_state_dict(checkpoint_t).items() if check_keep(k, _ignorekeywordlist)})

        _tmp_st = fusion_checkpoint(_tmp_st_rgb, _tmp_st_t)
        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))
    
    if cfg.EXPERIMENT.action == 'test':
        res, res_rgb, res_t, res_distill = evaluate(model, postprocessors, dataLoader, device, flops, cfg=cfg, distill=cfg.DISTILL.flag, info_weights=cfg.MODEL.MS_DETR.analysis_weights)

        return_result = list()
        if cfg.DISTILL.flag:
            print('=====DISTILL Branch Start=====')
            if cfg.TEST.metric == 'ap':
                ap_eval(res_distill)
            elif cfg.TEST.metric == 'cvc14':
                cvc14_eval(res_distill, cfg.EXPERIMENT.output_dir, resume, 'distill_fusion')
            else:
                return_result.append(kaist_eval(res_distill, cfg.EXPERIMENT.output_dir, resume, 'distill_fusion')[0])
            print('=====DISTILL Branch End======= \n \n')
        else:
            if res:
                print('=====Main Branch Start=====')
                if cfg.TEST.metric == 'ap':
                    ap_eval(res)
                elif cfg.TEST.metric == 'cvc14':
                    cvc14_eval(res, cfg.EXPERIMENT.output_dir, resume, 'fusion_branch')
                else:
                    return_result.append(kaist_eval(res, cfg.EXPERIMENT.output_dir, resume, 'fusion_branch', info_attention=cfg.MODEL.MS_DETR.analysis_weights)[0])
                print('=====Main Branch End======= \n \n')
            # if cfg.MODEL.MS_DETR.rgb_branch and res_rgb:
            #     print('=====RGB Branch Start=====')
            #     if cfg.TEST.metric == 'ap':
            #         ap_eval(res_rgb)
            #     elif cfg.TEST.metric == 'cvc14':
            #         cvc14_eval(res_rgb, cfg.EXPERIMENT.output_dir, resume, 'rgb_branch')
            #     else:
            #         return_result.append(kaist_eval(res_rgb, cfg.EXPERIMENT.output_dir, resume, 'rgb_fusion')[0])
            #     print('=====RGB Branch End======= \n \n')
            # if cfg.MODEL.MS_DETR.t_branch and res_t:
            #     print('=====Thermal Branch Start=====')
            #     if cfg.TEST.metric == 'ap':
            #         ap_eval(res_t)
            #     elif cfg.TEST.metric == 'cvc14':
            #         cvc14_eval(res_t, cfg.EXPERIMENT.output_dir, resume, 'thermal_branch')
            #     else:
            #         return_result.append(kaist_eval(res_t, cfg.EXPERIMENT.output_dir, resume, 'thermal_fusion')[0])
            #     print('=====Thermal Branch End======= \n \n')
        
        write_result_file(return_result, cfg.EXPERIMENT.output_dir, resume)
        return return_result

    if cfg.EXPERIMENT.action == 'cam':
        from visualize import detr_cam
        model.eval()
        detr_cam(model, dataset, cfg.EXPERIMENT.output_dir)
        return

    print("----------------------------Start training----------------------------")
    start_time = time.time()

    rgb_prototypes = None
    t_prototypes = None
    fusion_prototypes = None

    for epoch in range(cfg.EXPERIMENT.start_epoch, cfg.TRAIN.epochs):
        if cfg.MODEL.PROTOTYPE.flag:
            prototype_start_time = time.time()
            print("---------------------------------rank:{},更新Prototype--------------------------".format(cfg.DISTRIBUTED.rank))

            rgb_prototypes, t_prototypes, fusion_prototypes = cal_prototype(dataLoader, model, criterion.matcher, device, epoch, cfg, rgb_prototypes, t_prototypes, fusion_prototypes)
            prototype_time = time.time() - prototype_start_time
            total_prototype_time_str = str(datetime.timedelta(seconds=int(prototype_time)))

            print("---------------------------------rank:{}, Prototype更新结束, 耗时{}--------------------------".format(cfg.DISTRIBUTED.rank, total_prototype_time_str))

        epoch_start_time = time.time()
        if epoch == cfg.EXPERIMENT.close_strong_augment:
            dataLoader.dataset.update_skip_type_keys(["Mosaic", "RandomAffine", "MixUp", "RandAugment"])
            print('----------------------Close strong augmentation--------------------------------')

        if cfg.DISTRIBUTED.distributed:
            dataSampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, dataLoader, optimizer, device, epoch, cfg.TRAIN.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, cfg=cfg, logger=(logger if cfg.EXPERIMENT.save_log else None), rgb_prototypes=rgb_prototypes, t_prototypes=t_prototypes, fusion_prototypes=fusion_prototypes, distill=cfg.DISTILL.flag)

        checkpoint_paths = [output_dir / 'checkpoint.pth']
        # extra checkpoint before LR drop and every 100 epochs
        save_flag = False
        for lr_drop in cfg.TRAIN.lr_drop:
            if (epoch + 1) % lr_drop == 0:
                save_flag = True

        if save_flag:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}_beforedrop.pth')

        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'cfg': cfg,
            }, checkpoint_path)

        lr_scheduler.step()

        checkpoint_paths = [output_dir / 'checkpoint.pth']
        # extra checkpoint before LR drop and every 100 epochs

        save_flag = False
        for lr_drop in cfg.TRAIN.lr_drop:
            if (epoch + 1) % lr_drop == 0:
                save_flag = True

        if save_flag or (epoch + 1) % cfg.TRAIN.save_checkpoint_interval == 0:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'cfg': cfg,
            }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("Now time: {}".format(str(datetime.datetime.now())))


def fusion_checkpoint(orderDict_rgb, orderDict_t):
    loadMaps = OrderedDict()
    for key in orderDict_rgb.keys():
        if 'cross_attn.sampling_offsets.weight' in key:
            v_rgb = orderDict_rgb[key]
            v_t = orderDict_t[key]
            v_rgb = v_rgb.view(8, 4, 4, 2, 256)
            v_t = v_t.view(8, 4, 4, 2, 256)

            v = torch.cat([v_rgb, v_t], dim=1).view(512, 256)
            loadMaps[key] = v
        elif 'cross_attn.sampling_offsets.bias' in key:
            v_rgb = orderDict_rgb[key]
            v_t = orderDict_t[key]
            v_rgb = v_rgb.view(8, 4, 4, 2)
            v_t = v_t.view(8, 4, 4, 2)
            v = torch.cat([v_rgb, v_t], dim=1).view(512)
            loadMaps[key] = v
        elif 'cross_attn.attention_weights.weight' in key:
            v_rgb = orderDict_rgb[key]
            v_t = orderDict_t[key]
            v_rgb = v_rgb.view(8, 4, 4, 1, 256)
            v_t = v_t.view(8, 4, 4, 1, 256)
            v = torch.cat([v_rgb, v_t], dim=1).view(256, 256)
            loadMaps[key] = v
        elif 'cross_attn.attention_weights.bias' in key:
            v_rgb = orderDict_rgb[key]
            v_t = orderDict_t[key]
            v_rgb = v_rgb.view(8, 4, 4, 1)
            v_t = v_t.view(8, 4, 4, 1)
            v = torch.cat([v_rgb, v_t], dim=1).view(256)
            loadMaps[key] = v
        elif 'decoder' in key or 'transformer.reference_points' in key or 'query_embed' in key or 'class_embed' in key or 'bbox_embed' in key:
            v_rgb = orderDict_rgb[key]
            v_t = orderDict_t[key]
            v = (v_rgb + v_t) / 2
            loadMaps[key] = v
        else:
            v_rgb = orderDict_rgb[key]
            v_t = orderDict_t[key.replace('rgb', 't')]
            loadMaps[key] = v_rgb
            loadMaps[key.replace('rgb', 't')] = v_t

    return loadMaps


def getKeyMap4Distill(checkpoint, distill_modality_rgb, distill_rec):
    keyMap = dict()
    for k in checkpoint.keys():
        keyMap[k] = k  # 教师模型的参数直接加载
        if 'decoder' in k:
            if 'output_proj_rgb' in k:
                if distill_modality_rgb:
                    keyMap[k.replace('decoder', 'decoder_distill').replace('output_proj_rgb', 'output_proj')] = k  # 对于教师模型中的decoder中的output_proj_rgb，需要加载到decoder_distill中的output_proj（当蒸馏可见光模态时）
            elif 'output_proj_t' in k:
                if not distill_modality_rgb:
                    keyMap[k.replace('decoder', 'decoder_distill').replace('output_proj_t', 'output_proj')] = k  # 对于教师模型中的decoder中的output_proj_t，需要加载到decoder_distill中的output_proj（当蒸馏红外模态时）
            elif 'bbox_embed' in k:  # 对于教师模型中的decoder中的bbox_embed，不用迁移到学生模型中的decoder_distill，因为前者对应于融合特征
                pass
            else:
                keyMap[k.replace('decoder', 'decoder_distill')] = k  # 其余教师模型中的decoder中的参数全部复制到学生模型中的decoder_distill
        elif 'bbox_embed' in k and not 'bbox_embed_rgb' in k and not 'bbox_embed_t' in k:
            if distill_rec:
                keyMap[k.replace('bbox_embed', 'bbox_embed_distill')] = k  # 教师模型中的bbox_embed需要复制到两个地方，一个是MS_DETR的bbox_embed_distill中，另外一个是decoder_distill中的bbox_embed
                keyMap['transformer.decoder_distill.' + k.replace('bbox_embed', 'bbox_embed')] = k
        elif 'bbox_embed_rgb' in k:
            if distill_modality_rgb and not distill_rec:
                keyMap[k.replace('bbox_embed_rgb', 'bbox_embed_distill')] = k  # 教师模型中的bbox_embed_rgb需要复制到两个地方，一个是MS_DETR的bbox_embed_distill中，另外一个是decoder_distill中的bbox_embed
                keyMap['transformer.decoder_distill.' + k.replace('bbox_embed_rgb', 'bbox_embed')] = k
        elif 'bbox_embed_t' in k:
            if not distill_modality_rgb and not distill_rec:
                keyMap[k.replace('bbox_embed_t', 'bbox_embed_distill')] = k
                keyMap['transformer.decoder_distill.' + k.replace('bbox_embed_t', 'bbox_embed')] = k
        elif 'class_embed' in k and not 'class_embed_rgb' in k and not 'class_embed_t' in k:
            if distill_rec:
                keyMap[k.replace('class_embed', 'class_embed_distill')] = k
        elif 'class_embed_rgb' in k:
            if distill_modality_rgb and not distill_rec:
                keyMap[k.replace('class_embed_rgb', 'class_embed_distill')] = k
        elif 'class_embed_t' in k:
            if not distill_modality_rgb and not distill_rec:
                keyMap[k.replace('class_embed_t', 'class_embed_distill')] = k
    return keyMap


def getKeyMap(checkpoint, split_cls_reg):
    keyMap = dict()
    for k in checkpoint.keys():
        if 'level_embed' in k:
            keyMap[k.replace('level_embed', 'level_embed_rgb')] = k
            keyMap[k.replace('level_embed', 'level_embed_t')] = k
            keyMap[k.replace('level_embed', 'level_embed_fusion')] = k
            keyMap[k.replace('level_embed', 'level_embed_share')] = k
        elif 'encoder' in k:
            keyMap[k.replace('encoder', 'encoder_rgb')] = k
            keyMap[k.replace('encoder', 'encoder_t')] = k
            keyMap[k.replace('encoder', 'encoder_fusion')] = k
            keyMap[k.replace('encoder', 'encoder_share')] = k
            keyMap[k.replace('encoder', 'encoder_4_fusion')] = k
        elif 'body' in k:
            keyMap[k.replace('body', 'body_rgb')] = k
            keyMap[k.replace('body', 'body_t')] = k
        elif 'input_proj' in k:
            # pass
            keyMap[k.replace('input_proj', 'input_proj_rgb')] = k
            keyMap[k.replace('input_proj', 'input_proj_t')] = k
            keyMap[k.replace('input_proj', 'input_proj_fusion')] = k
            keyMap[k.replace('input_proj', 'input_proj_share')] = k
        elif 'decoder' not in k and 'bbox_embed' in k and not split_cls_reg:
            keyMap[k] = k
            keyMap[k.replace('bbox_embed', 'bbox_embed_rgb')] = k
            keyMap[k.replace('bbox_embed', 'bbox_embed_t')] = k
        elif 'decoder' in k:
            if 'cross_attn.output_proj' in k:
                keyMap[k] = k
                keyMap[k.replace('cross_attn.output_proj', 'cross_attn.output_proj_rgb')] = k
                keyMap[k.replace('cross_attn.output_proj', 'cross_attn.output_proj_t')] = k
            elif 'class_embed' in k:
                pass
            elif 'bbox_embed' in k:
                if not split_cls_reg:
                    keyMap[k] = k
                    keyMap[k.replace('bbox_embed', 'bbox_embed_rgb')] = k
                    keyMap[k.replace('bbox_embed', 'bbox_embed_t')] = k
            else:
                keyMap[k] = k
        elif 'reference_points' in k or 'enc_output' in k or 'pos_trans' in k:
            keyMap[k] = k
        elif 'query_embed' in k:
            keyMap[k] = k
        else:
            print(k, checkpoint[k].shape)
    return keyMap


def write_result_file(result_list, output_dir, resume):
    if len(result_list):
        import fcntl  

        output_dir = os.path.join(output_dir, 'test')
        checkpoint = os.path.basename(resume).split('.')[0]

        file_path = os.path.join(output_dir, 'all_result.txt')

        with open(file_path, 'a') as file:
            fcntl.flock(file, fcntl.LOCK_EX)

            data = str(checkpoint) + '  :  '
            for r in result_list:
                data += '  '.join(f'{num * 100:.2f}' for num in r)
                data += '  |  '

            data += '\n'

            try:
                # 将数据写入文件
                file.write(data)
            finally:
                # 释放文件锁
                fcntl.flock(file, fcntl.LOCK_UN)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('MS-DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices

    cfg = get_cfg_defaults()
    if args.exp_config:
        cfg.merge_from_file(args.exp_config)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    os.environ['output_dir'] = args.output_dir

    cfg.EXPERIMENT.action = args.action
    cfg.EXPERIMENT.output_dir = args.output_dir

    main(args.output_dir, args.resume, args.find_unused_params, args.flops, args.only_fusion, cfg)
