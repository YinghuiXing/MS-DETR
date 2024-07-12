import copy
import os.path

import torch
import collections
import datasets.transforms as T
import shutil
from torch.utils.data.dataset import Dataset as torchDataset
from visualize.vis_tools import drawBBoxes
from util.box_ops import box_cxcywh_to_xyxy


def visualize_masks_decoder(masks, bboxes, img):
    for ind, (mask, box) in enumerate(zip(masks, bboxes)):
        mask_saved_path = f'visualize_masks/mask_{ind}.jpg'
        img_saved_path = f'visualize_masks/img_{ind}.jpg'
        drawBBoxes(mask, [box], widths=4.0, bboxes_mode='xywh', saved_path=mask_saved_path, percentile=True)
        drawBBoxes((img.permute(1, 2, 0) * 255).to(torch.uint8), [box], widths=4.0, bboxes_mode='xywh', saved_path=img_saved_path, percentile=True)


def visualize_masks(masks, bboxes, img):
    mask_saved_path = f'visualize_masks/mask_all.jpg'
    img_saved_path = f'visualize_masks/img_all.jpg'
    drawBBoxes(masks, bboxes, bboxes_mode='xywh', saved_path=mask_saved_path, percentile=True)
    drawBBoxes((img.permute(1, 2, 0) * 255).to(torch.uint8), bboxes, widths=4.0, bboxes_mode='xywh', saved_path=img_saved_path)


def visualize_transform(results, transform_type, transform_ind):

    rgb_img = results['rgb_img'] if 'rgb_img' in results else None
    t_img = results['t_img'] if 't_img' in results else None

    gt_boxes_rgb = results['gt_bboxes_rgb'] if 'gt_bboxes_rgb' in results else None
    gt_boxes_t = results['gt_bboxes_t'] if 'gt_bboxes_t' in results else None

    mix_results = results['mix_results'] if 'mix_results' in results else None
    if mix_results is not None:
        ind = 0
        for mix_result in mix_results:
            mix_rgb_img = mix_result['rgb_img'] if 'rgb_img' in mix_result else None
            mix_t_img = mix_result['t_img'] if 't_img' in mix_result else None
            mix_gt_boxes_rgb = mix_result['gt_bboxes_rgb'] if 'gt_bboxes_rgb' in mix_result else None
            mix_gt_boxes_t = mix_result['gt_bboxes_t'] if 'gt_bboxes_t' in mix_result else None

            if mix_rgb_img is not None:
                mix_rgb_img_saved_path = f'visualize_transform/{transform_ind}_{transform_type}/{transform_type}_mix_rgb_{ind}.tif'
                drawBBoxes(mix_rgb_img, mix_gt_boxes_rgb, bboxes_mode='xyxy', saved_path=mix_rgb_img_saved_path)

            if mix_t_img is not None:
                mix_t_img_saved_path = f'visualize_transform/{transform_ind}_{transform_type}/{transform_type}_mix_t_{ind}.tif'
                drawBBoxes(mix_t_img, mix_gt_boxes_t, bboxes_mode='xyxy', saved_path=mix_t_img_saved_path)
            ind += 1

    if rgb_img is not None:
        rgb_img_saved_path = f'visualize_transform/{transform_ind}_{transform_type}/After_{transform_type}_rgb.tif'
        if isinstance(rgb_img, torch.Tensor):
            rgb_img = (rgb_img.permute(1, 2, 0) * 255).to(torch.uint8)
            drawBBoxes(rgb_img, gt_boxes_rgb, percentile=True, saved_path=rgb_img_saved_path)
        else:
            drawBBoxes(rgb_img, gt_boxes_rgb, bboxes_mode='xyxy', saved_path=rgb_img_saved_path)
        print(f'After {transform_type}, the num of bbox in rgb modality is {len(gt_boxes_rgb)}')
        print(f'After {transform_type}, the shape of img in rgb modality is {rgb_img.shape}')

    if t_img is not None:
        t_img_saved_path = f'visualize_transform/{transform_ind}_{transform_type}/After_{transform_type}_t.tif'
        if isinstance(t_img, torch.Tensor):
            t_img = (t_img.permute(1, 2, 0) * 255).to(torch.uint8)
            drawBBoxes(t_img, gt_boxes_t, percentile=True, saved_path=t_img_saved_path)
        else:
            drawBBoxes(t_img, gt_boxes_t, bboxes_mode='xyxy', saved_path=t_img_saved_path)
        print(f'After {transform_type}, the num of bbox in thermal modality is {len(gt_boxes_rgb)}')
        print(f'After {transform_type}, the shape of img in thermal modality is {t_img.shape}\n')


def make_transforms(action, transform_cfg):
    img_scale = transform_cfg.img_size
    img_resize_sizes = transform_cfg.img_resize_sizes

    img_resize_sizes = [tuple(_) for _ in img_resize_sizes]
    flip_ratio = transform_cfg.flip_ratio
    mosaic_flag = transform_cfg.mosaic_flag
    mix_up_flag = transform_cfg.mix_up_flag
    random_affine_flag = transform_cfg.random_affine_flag

    transforms = list()
    if action == 'train':
        if mosaic_flag:
            if transform_cfg.random_affine_before_mosaic_flag:
                transforms.append(T.Mosaic(img_scale=img_scale, center_ratio_range=(0.5, 1.5), random_affine_engine=T.RandomAffine(border=(-img_scale[0] // 4, -img_scale[1] // 4))))
            else:
                transforms.append(T.Mosaic(img_scale=img_scale, center_ratio_range=(0.5, 1.5)))

        if random_affine_flag:
            transforms.append(T.RandomAffine(border=(-img_scale[0] // 2, -img_scale[1] // 2)))

        if transform_cfg.RANDAUGMENT.flag:
            transforms.append(T.RandAugment(aug_space=transform_cfg.RANDAUGMENT.aug_space,
                                            aug_num=transform_cfg.RANDAUGMENT.num,
                                            level=transform_cfg.RANDAUGMENT.level,
                                            random=transform_cfg.RANDAUGMENT.random))
        if mix_up_flag:
            transforms.append(T.MixUp(img_scale=img_scale))

        transforms.append(T.RandomFlip(flip_ratio=flip_ratio))
        # transforms.append(T.Resize(img_scale=img_scale, ratio_range=(1.68, 1.75)))
        transforms.append(T.Resize(img_scale=img_resize_sizes, multiscale_mode='value'))
        transforms.append(T.FilterAnnotations(min_gt_bbox_wh=transform_cfg.min_gt_bbox_wh, keep_empty=False))
        transforms.append(T.DefaultFormatBundle())
    elif action in ('test', 'cam', 'inference'):
        # transforms.append(T.RandomFlip(flip_ratio=0.5))
        # transforms.append(T.Resize(img_scale=img_scale, ratio_range=(1.68, 1.75)))
        transforms.append(T.Resize(img_scale=img_resize_sizes, multiscale_mode='value'))
        transforms.append(T.DefaultFormatBundle())

    return transforms


class MultiImageMixDataset(torchDataset):
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process. At the same time, we provide the `dynamic_scale` parameter
    to dynamically change the output image size.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be mixed.
        transforms (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        skip_type_keys
    """

    def __init__(self, dataset, skip_type_keys=None, mask=False, just_test=False, transforms_cfg=None):
        super(MultiImageMixDataset, self).__init__()
        self._transforms = make_transforms(dataset.action, transforms_cfg)
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.action = dataset.action
        self.mask = mask

        if skip_type_keys is not None:
            assert all([isinstance(skip_type_key, str) for skip_type_key in skip_type_keys])
        self._skip_type_keys = skip_type_keys
        self.just_test = just_test

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        while True:
            break_flag = False  # 数据增强后存在行人目标才能够退出

            results = copy.deepcopy(self.dataset[idx])

            if self.just_test and os.path.isdir('visualize_transform'):
                shutil.rmtree('visualize_transform')

            for transform_ind, transform in enumerate(self._transforms):
                transform_type = type(transform).__name__

                if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                    continue

                if hasattr(transform, 'get_indexes'):
                    indexes = transform.get_indexes(self.dataset, idx)

                    if not isinstance(indexes, collections.abc.Sequence):
                        indexes = [indexes]

                    mix_results = [
                        copy.deepcopy(self.dataset[index]) for index in indexes
                    ]
                    results['mix_results'] = mix_results

                results = transform(results)

                if self.just_test:
                    visualize_transform(results, transform_type, transform_ind)

                if 'mix_results' in results:
                    results.pop('mix_results')

            if self.action != 'train':
                break_flag = True
            else:
                for key in results['bbox_fields']:
                    if len(results[key]) > 0:
                        break_flag = True

            if break_flag:
                break

        target = dict()
        target['orig_size'] = torch.as_tensor(results['ori_shape'][:2])
        target['size'] = torch.as_tensor(results['rgb_img'].shape[1:]) if 'rgb_img' in results else torch.as_tensor(results['t_img'].shape[1:])

        for key in results['label_fields']:
            target[key] = results[key]
        for key in results['bbox_fields']:
            import numpy
            if isinstance(results[key], numpy.ndarray):
                target[key] = torch.tensor(results[key]).to(torch.float32)
            else:
                target[key] = results[key].to(torch.float32)

        if self.action in ['test', 'cam', 'inference']:
            target['image_ind'] = results['image_ind']
            target['img_absolute_path_rgb'] = results['img_absolute_path_rgb']
            target['anno_absolute_path_rgb'] = results['anno_absolute_path_rgb']
            target['img_absolute_path_t'] = results['img_absolute_path_t']
            target['anno_absolute_path_t'] = results['anno_absolute_path_t']
            target['anno_absolute_path'] = results['anno_absolute_path_rgb']

            target['flip'] = results['flip'] if 'flip' in results else None
            target['flip_direction'] = results['flip_direction'] if 'flip_direction' in results else None

        if self.mask:
            masks_decoder = self._generate_masks_4_seg_decoder(target)
            masks = self._generate_masks_4_seg(target)

            if self.just_test:
                visualize_masks_decoder(masks_decoder, target['gt_bboxes_rgb'], results['rgb_img'])
                visualize_masks(masks, target['gt_bboxes_rgb'], results['rgb_img'])

            target['masks_decoder'] = masks_decoder
            target['masks'] = masks

        return results['rgb_img'] if 'rgb_img' in results else None, results['t_img'] if 't_img' in results else None, target

    def _generate_masks_4_seg_decoder(self, target):
        h, w = target['size']
        bboxes = target['gt_bboxes_rgb']
        bboxes = box_cxcywh_to_xyxy(bboxes)
        bboxes = (bboxes * torch.tensor([w, h, w, h], dtype=torch.float32)).to(torch.int)
        masks = torch.zeros((bboxes.shape[0], h, w), dtype=torch.bool, device=bboxes.device)
        for ind, box in enumerate(bboxes):
            masks[ind, box[1]:box[3], box[0]:box[2]] = True

        return masks

    def _generate_masks_4_seg(self, target):
        h, w = target['size']
        bboxes = target['gt_bboxes_rgb']
        bboxes = box_cxcywh_to_xyxy(bboxes)
        bboxes = (bboxes * torch.tensor([w, h, w, h], dtype=torch.float32)).to(torch.int)
        masks = torch.zeros((h, w), dtype=torch.bool, device=bboxes.device)
        for box in bboxes:
            masks[box[1]:box[3], box[0]:box[2]] = True

        return masks


    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys
