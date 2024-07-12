# -*- coding: utf-8 -*-
# @Time    : 2023/10/25 16:10
# @Author  : WangSong
# @Email   : 1021707198@qq.com
# @File    : prototype.py

import torch
from torch import nn
import torch.distributed as dist

class Prototype(nn.Module):
    def __init__(self, num_classes, hidden_dim, momentum_coef):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.momentum_coef = momentum_coef
        self.rgb_prototypes = torch.zeros(num_classes, hidden_dim)
        self.t_prototypes = torch.zeros(num_classes, hidden_dim)
        self.fusion_prototypes = torch.zeros(num_classes, hidden_dim)

    def forward(self, dataloader, model, matcher, device, epoch, cfg):
        rgb_prototypes = torch.zeros(self.num_classes, self.hidden_dim)
        t_prototypes = torch.zeros(self.num_classes, self.hidden_dim)
        fusion_prototypes = torch.zeros(self.num_classes, self.hidden_dim)

        count_class = [0 for _ in range(self.num_classes)]

        model.eval()
        with torch.no_grad:
            sample_count = 0
            all_num = len(dataloader)

            for *samples, targets in dataloader:
                samples = [(item.to(device) if item is not None else None) for item in samples]

                # 混合精度训练
                with torch.cuda.amp.autocast(enabled=cfg.EXPERIMENT.amp):
                    outputs, outputs_rgb, outputs_t = model(*samples, targets)
               
                hs = outputs['hs']
                hs_rgb = outputs_rgb['hs_rgb']
                hs_t = outputs_t['hs_t']

                last_hs = hs[-1]
                last_hs_rgb = hs_rgb[-1]
                last_hs_t = hs_t[-1]

                outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

                if outputs_rgb is not None and outputs_t is not None:
                    outputs_rgb_without_aux = {k: v for k, v in outputs_rgb.items() if k != 'aux_outputs' and k != 'enc_outputs'}
                    outputs_t_without_aux = {k: v for k, v in outputs_t.items() if k != 'aux_outputs' and k != 'enc_outputs'}
                else:
                    outputs_rgb_without_aux = None
                    outputs_t_without_aux = None
                
                # indices的样例：[(tensor([130, 271]), tensor([0, 1])), (tensor([  8,  18,  66,  68, 147]), tensor([3, 2, 0, 4, 1]))]
                indices, _, __, ___ = matcher(outputs_without_aux, targets, outputs_rgb_without_aux, outputs_t_without_aux)
                for i, (indice, target) in enumerate(zip(indices, targets)):
                    positive_oq_feature = last_hs[indice[0]]
                    positive_oq_feature_rgb = last_hs_rgb[indice[0]]
                    positive_oq_feature_t = last_hs_t[indice[0]]

                    labels = target[cfg.MODEL.LOSS.gt_field_class][indice[1]]

                    for j, (feature, feature_rgb, feature_t, label) in enumerate(positive_oq_feature, positive_oq_feature_rgb, positive_oq_feature_t, labels):
                        fusion_prototypes[label, :] += feature
                        rgb_prototypes[label, :] += feature_rgb
                        t_prototypes[label, :] += feature_t
                        count_class += 1
                
                sample_count += 1
                if sample_count >= all_num // 10:
                    break

            for c in range(self.num_classes):
                fusion_prototypes[c, :] /= count_class[c]
                rgb_prototypes[c, :] /= count_class[c]
                t_prototypes[c, :] /= count_class[c]
            
            dist.all_reduce(fusion_prototypes)
            dist.all_reduce(rgb_prototypes)
            dist.all_reduce(t_prototypes)

            if epoch <= 0:
                self.fusion_prototypes = fusion_prototypes
                self.rgb_prototypes = rgb_prototypes
                self.t_prototypes = t_prototypes
            else:
                self.fusion_prototypes = (1 - self.momentum_coef) * fusion_prototypes + self.momentum_coef * self.fusion_prototypes
                self.rgb_prototypes = (1 - self.momentum_coef) * rgb_prototypes + self.momentum_coef * self.rgb_prototypes
                self.t_prototypes = (1 - self.momentum_coef) * t_prototypes + self.momentum_coef * self.t_prototypes


                







                




