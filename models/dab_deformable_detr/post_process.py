# -*- coding: utf-8 -*-
# @Time    : 2023/2/12 15:36
# @Author  : WangSong
# @Email   : 1021707198@qq.com
# @File    : post_process.py
import torch
from torch import nn

from util import box_ops


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, cfg) -> None:
        super().__init__()
        self.num_select = cfg.num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes, flip, info_weights=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            flip: 存放图像翻转的方向，horizontal|vertical|diagonal|None
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        if info_weights:
            out_weights_rgb = outputs['weights_rgb']
            out_weights_t = outputs['weights_t']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        assert len(flip) == len(out_bbox)

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values # shape为2*300
        topk_boxes = topk_indexes // out_logits.shape[2] # shape为2*300,其元素的含义是按照得分排列的检测结果的下标，故而元素的范围为[0, 299]
        labels = topk_indexes % out_logits.shape[2]  # shape为2*300, 其元素的含义是按照得分排列的检测结果的标签，例如有两类的话，其取值范围就是0,1

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))  # topk_boxes.unsqueeze(-1).repeat(1,1,4)形状为2*300*4

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        for i in range(len(flip)):
            direction = flip[i]
            image_shape = target_sizes[i]
            boxes[i] = self.bbox_flip(boxes[i], image_shape, direction)
        
        if info_weights:
            for i, (ind, weights_rgb, weights_t) in enumerate(zip(topk_boxes, out_weights_rgb, out_weights_t)):
                out_weights_rgb[i] = weights_rgb[ind]
                out_weights_t[i] = weights_t[ind]

        if info_weights:
            results = [{'scores': s, 'labels': l, 'boxes': b, 'attention_rgb': ar, 'attention_t': at} for s, l, b, ar, at in zip(scores, labels, boxes, out_weights_rgb, out_weights_t)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.clone()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction is None:
            pass
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped