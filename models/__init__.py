# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from .dab_deformable_detr import build_dab_deformable_detr


def build_model(only_fusion, cfg, cfg_distill, cfg_rec):
    if cfg.detr_name.lower() == 'deformable_detr':
        model, criterion, postprocessors = build_dab_deformable_detr(only_fusion, cfg, cfg_distill, cfg_rec)
    else:
        raise NotImplementedError

    return model, criterion, postprocessors

