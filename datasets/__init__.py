# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data

import util.misc as utils
import datasets.transforms as T
from datasets.kaist import KaistDetection
from datasets.kaist_voc import KAISTVOCDetection
from datasets.dataset_wrappers import MultiImageMixDataset


def build_kaist_or_voc_dataset(action, mask, format, cfg, transforms_cfg):
    dataset = KaistDetection(cfg.root_dirs_train if action == 'train' else cfg.root_dirs_test,
                             cfg.rgb_datasets_train if action == 'train' else cfg.rgb_datasets_test,
                             cfg.t_datasets_train if action=='train' else cfg.t_datasets_test, action,
                             filter_mode=cfg.filter_mode, cut_out_filter=cfg.cut_out_filter,
                             gt_merge=cfg.gt_merge, gt_merge_mode=cfg.gt_merge_mode, cvc14=(format == 'cvc14'))

    AugDataset = MultiImageMixDataset(dataset, mask=mask, transforms_cfg=transforms_cfg)

    return AugDataset



def build_dataset(action, mask, cfg):
    dataset_format = cfg.format

    if dataset_format in ('kaist', 'voc', 'cvc14'):
        dataset = build_kaist_or_voc_dataset(action, mask, dataset_format, cfg.KAIST, cfg.TRANSFORMS)
        return dataset
    else:
        raise ValueError(f'dataset {cfg.format} not supported')


def build_dataLoader(cfg):
    action = cfg.EXPERIMENT.action
    mask = cfg.MODEL.MS_DETR.SEGMENTATION.flag
    dataset = build_dataset(action, mask, cfg.DATASET)

    if cfg.DISTRIBUTED.distributed:
        sampler = torch.utils.data.DistributedSampler(dataset)
    else:
        if action == 'train':
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    # def seed_worker(worker_id):
    #     worker_seed = torch.initial_seed() % 2 ** 32
    #     numpy.random.seed(worker_seed)
    #     random.seed(worker_seed)
    #
    # g = torch.Generator()
    # g.manual_seed(0)

    # If train, the sampler need become BatchSampler
    batch_sampler = None
    if action == 'train':
        batch_sampler = torch.utils.data.BatchSampler(sampler, cfg.DATASET.batch_size, drop_last=True)
        # dataLoader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=utils.collate_fn,
        #                                          num_workers=cfg.DATASET.num_workers, worker_init_fn=seed_worker,
        #                                          generator=g)
        dataLoader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=utils.collate_fn,
                                                 num_workers=cfg.DATASET.num_workers)
    else:
        # dataLoader = torch.utils.data.DataLoader(dataset, cfg.DATASET.batch_size, sampler=sampler, drop_last=False,
        #                                          collate_fn=utils.collate_fn, num_workers=cfg.DATASET.num_workers,
        #                                          worker_init_fn=seed_worker, generator=g)
        dataLoader = torch.utils.data.DataLoader(dataset, cfg.DATASET.batch_size, sampler=sampler, drop_last=False,
                                                 collate_fn=utils.collate_fn, num_workers=cfg.DATASET.num_workers)

    return dataset, sampler, batch_sampler, dataLoader