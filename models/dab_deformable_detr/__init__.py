import torch
from .backbone import build_backbone
from .deformable_transformer import DeformableTransformer
from .dab_deformable_detr import MS_DETR
from .loss import SetCriterion
from .matcher import HungarianMatcher
from .post_process import PostProcess
from .prototype import Prototype


def build_dab_deformable_detr(only_fusion, cfg, cfg_distill, cfg_rec):
    device = torch.device(cfg.device)

    rgb_branch = cfg.MS_DETR.rgb_branch
    t_branch = cfg.MS_DETR.t_branch
    gt_field_class = cfg.LOSS.gt_field_class
    gt_field_bbox = cfg.LOSS.gt_field_bbox

    distill = cfg_distill.flag
    distill_modality_rgb = cfg_distill.distill_modality_rgb
    follow_teacher = cfg_distill.follow_teacher
    distill_inter_references = cfg_distill.distill_inter_references
    distill_features = cfg_distill.distill_features
    distill_fusion_features = cfg_distill.distill_fusion_features

    rec_flag = cfg_rec.flag
    rec_momdality_rgb = cfg_rec.momdality_rgb
    rec_features_loss = cfg_rec.features_loss
    rec_features_loss_coef = cfg_rec.features_loss_coef
    rec_freeze = cfg_rec.freeze

    rec_fusion = cfg_distill.rec_fusion
    rec_another = cfg_distill.rec_another
    rec_use_kernal_3 = cfg_distill.rec_use_kernal_3
    
    assert not (rec_fusion and rec_another)

    backbone = build_backbone(cfg.BACKBONE, cfg.POSITION_ENCODING)
    if distill or (rec_flag and rec_freeze):
        for param in backbone.parameters():
            param.requires_grad_(False)
    
    transformer = DeformableTransformer(cfg.MS_DETR.use_dab, rgb_branch, t_branch, cfg.MS_DETR.modality_crossover, cfg.MS_DETR.modality_decoupled, 
                                        cfg.MS_DETR.two_stage, cfg.MS_DETR.num_queries, cfg.MS_DETR.split_cls_reg, cfg.TRANSFORMER, only_fusion, 
                                        distill, distill_modality_rgb, cfg_distill.DECODER, rec_another,
                                        rec_flag, rec_momdality_rgb, rec_freeze)

    model = MS_DETR(backbone, transformer, cfg.MS_DETR, cfg.num_classes, cfg.LOSS.aux_loss, 
                    distill, distill_modality_rgb, distill_fusion_features, rec_fusion, rec_another, rec_use_kernal_3,
                    rec_flag, rec_freeze)

    matcher = HungarianMatcher(rgb_branch, t_branch, cfg.LOSS.instance_reweight, cfg.MS_DETR.modality_decoupled, gt_field_class, gt_field_bbox, cfg.MATCHER,
                               cfg.LOSS.reweight_hard, cfg.LOSS.positive_alpha, cfg.LOSS.negative_alpha, cfg.LOSS.adaptive_reweight, 
                               cfg.LOSS.plus_three, cfg.LOSS.use_p)

    weight_dict = {'loss_ce': cfg.LOSS.cls_loss_coef, 'loss_bbox': cfg.LOSS.bbox_loss_coef, 'loss_giou': cfg.LOSS.giou_loss_coef}
    if distill:
        if distill_inter_references:
            weight_dict['loss_distill_bbox'] = cfg_distill.distill_inter_references_bbox_loss_coef
            weight_dict['loss_distill_giou'] = cfg_distill.distill_inter_references_giou_loss_coef
        if distill_features:
            weight_dict['loss_distill_features'] = cfg_distill.distill_features_loss_coef

    if rec_flag:
        weight_dict['loss_rec_features'] = rec_features_loss_coef

    if cfg.PROTOTYPE.flag:
        weight_dict['loss_prototype'] = cfg.PROTOTYPE.loss_coef

    if cfg.MS_DETR.SEGMENTATION.flag:
        if cfg.MS_DETR.SEGMENTATION.stage == 'decoder':
            weight_dict["loss_mask_rgb"] = cfg.LOSS.mask_loss_coef
            weight_dict["loss_dice_rgb"] = cfg.LOSS.dice_loss_coef
            weight_dict["loss_mask_t"] = cfg.LOSS.mask_loss_coef
            weight_dict["loss_dice_t"] = cfg.LOSS.dice_loss_coef
        else:
            weight_dict["loss_mask_rgb"] = 1
            weight_dict["loss_mask_t"] = 1
            weight_dict["loss_mask_fusion"] = 1

    if cfg.LOSS.aux_loss:
        aux_weight_dict = {}
        for i in range(cfg.TRANSFORMER.DECODER.layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    if cfg.MS_DETR.SEGMENTATION.flag:
        if cfg.MS_DETR.SEGMENTATION.stage in ('backbone', 'encoder'):
            losses += ["masks"]
        else:
            losses += ['masks_decoder']
    
    if distill:
        if distill_inter_references:
            losses += ['distill_boxes']
        if distill_features:
            losses += ['distill_features']
    
    if rec_flag:
        losses += ['rec_features']

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(cfg.num_classes, matcher, weight_dict, losses, gt_field_class, gt_field_bbox, focal_alpha=cfg.LOSS.focal_alpha, instance_reweight=cfg.LOSS.instance_reweight, term_reweight=cfg.LOSS.term_reweight,
                             prototype=cfg.PROTOTYPE.flag, prototype_reweight=cfg.PROTOTYPE.reweight, prototype_all_layers=cfg.PROTOTYPE.all_layers, prototype_alpha=cfg.PROTOTYPE.alpha, follow_last_layer=cfg.LOSS.follow_last_layer, 
                             split_cls_reg=cfg.MS_DETR.split_cls_reg, adaptive_reweight=cfg.PROTOTYPE.adaptive_reweight, 
                             distill=distill, follow_teacher=follow_teacher, distill_features_loss=cfg_distill.distill_features_loss, rec_features_loss=rec_features_loss)
    criterion.to(device)

    postprocessors = {'bbox': PostProcess(cfg.POST_PROCESS)}

    return model, criterion, postprocessors



