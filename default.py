# -*- coding: utf-8 -*-
# @Time    : 2023/2/7 14:56
# @Author  : WangSong
# @Email   : 1021707198@qq.com
# @File    : default.py

from yacs.config import CfgNode as CN

_C = CN()

_C.TRAIN = CN()
_C.TRAIN.lr = 1e-4
_C.TRAIN.lr_backbone = 1e-5
_C.TRAIN.epochs = 60
_C.TRAIN.lr_drop = [40, 50]
_C.TRAIN.lr_drop_rate = 0.1
_C.TRAIN.weight_decay = 1e-4
_C.TRAIN.clip_max_norm = 0.1
_C.TRAIN.save_checkpoint_interval = 5

_C.TEST = CN()
_C.TEST.metric = 'mr^-2'

_C.MODEL = CN()
_C.MODEL.detr_name = 'deformable_detr'
_C.MODEL.detr_hidden_dim = 256
_C.MODEL.num_classes = 2
_C.MODEL.device = 'cuda'

_C.MODEL.MS_DETR = CN()
_C.MODEL.MS_DETR.num_queries = 100
_C.MODEL.MS_DETR.two_stage = False
_C.MODEL.MS_DETR.num_feature_levels = 3
_C.MODEL.MS_DETR.use_dab = True
_C.MODEL.MS_DETR.random_xy = False
_C.MODEL.MS_DETR.with_box_refine = True
_C.MODEL.MS_DETR.encoder_share = False
_C.MODEL.MS_DETR.content_embedding = True
_C.MODEL.MS_DETR.rgb_branch = False
_C.MODEL.MS_DETR.t_branch = False
_C.MODEL.MS_DETR.modality_crossover = False
_C.MODEL.MS_DETR.modality_decoupled = False
_C.MODEL.MS_DETR.share_object_queries = True  # 当三个分支共享oq时，object queries一致，并且object queries的更新也一致
_C.MODEL.MS_DETR.two_stage = False
_C.MODEL.MS_DETR.split_cls_reg = False  # 将object queries中的cls和reg分开
_C.MODEL.MS_DETR.analysis_weights = False
_C.MODEL.MS_DETR.SEGMENTATION = CN()
_C.MODEL.MS_DETR.SEGMENTATION.flag = False
_C.MODEL.MS_DETR.SEGMENTATION.freeze_detr = False
_C.MODEL.MS_DETR.SEGMENTATION.dropout = 0.1
_C.MODEL.MS_DETR.SEGMENTATION.stage = 'backbone'
_C.MODEL.MS_DETR.SEGMENTATION.vis = False
_C.MODEL.MS_DETR.SEGMENTATION.re_weight = False
_C.MODEL.MS_DETR.SEGMENTATION.re_weight_detach = False

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.backbone_rgb = 'resnet50'
_C.MODEL.BACKBONE.backbone_t = 'resnet50'
_C.MODEL.BACKBONE.backbone_share = False
_C.MODEL.BACKBONE.dilation = False  # If true, we replace stride with dilation in the last convolutional block (DC5)
_C.MODEL.BACKBONE.batch_norm_type = 'FrozenBatchNorm2d'  # choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d']
_C.MODEL.BACKBONE.train_layers_rgb = [2, 3, 4]
_C.MODEL.BACKBONE.train_layers_t = [2, 3, 4]
_C.MODEL.BACKBONE.return_layers = [2, 3, 4]
_C.MODEL.BACKBONE.strides = [2, 4, 8, 16, 32]
_C.MODEL.BACKBONE.num_channels_rgb = [64, 256, 512, 1024, 2048]  # resnet18, resnet34: [128, 256, 512], vgg16:[256, 512, 512]
_C.MODEL.BACKBONE.num_channels_t = [64, 256, 512, 1024, 2048]

_C.MODEL.BACKBONE.UP_SAMPLING = CN()
_C.MODEL.BACKBONE.UP_SAMPLING.up_sampling_flag = False
_C.MODEL.BACKBONE.UP_SAMPLING.up_sampling_stride = 8

_C.MODEL.BACKBONE.SHARE = CN()
_C.MODEL.BACKBONE.SHARE.share_flag = False
_C.MODEL.BACKBONE.SHARE.share_fusion_ind = 1
_C.MODEL.BACKBONE.SHARE.return_layers = [2, 3, 4]
_C.MODEL.BACKBONE.SHARE.num_channels_share = [64, 256, 512, 1024, 2048]
_C.MODEL.BACKBONE.SHARE.fusion_module = 'conv3x3'

_C.MODEL.BACKBONE.FEATURE_FUSION = CN()
_C.MODEL.BACKBONE.FEATURE_FUSION.fusion_flag = False
_C.MODEL.BACKBONE.FEATURE_FUSION.only_fusion = False
_C.MODEL.BACKBONE.FEATURE_FUSION.fusion_module = 'conv3x3'
_C.MODEL.BACKBONE.FEATURE_FUSION.num_channels_fusion = [64, 256, 512, 1024, 2048]

_C.MODEL.POSITION_ENCODING = CN()
_C.MODEL.POSITION_ENCODING.position_embedding = 'sine'
_C.MODEL.POSITION_ENCODING.pe_temperatureH = 20
_C.MODEL.POSITION_ENCODING.pe_temperatureW = 20
_C.MODEL.POSITION_ENCODING.N_steps = _C.MODEL.detr_hidden_dim // 2

_C.MODEL.TRANSFORMER = CN()
_C.MODEL.TRANSFORMER.dim_feedforward = 2048
_C.MODEL.TRANSFORMER.hidden_dim = _C.MODEL.detr_hidden_dim
_C.MODEL.TRANSFORMER.n_heads = 8
_C.MODEL.TRANSFORMER.num_feature_levels = _C.MODEL.MS_DETR.num_feature_levels
_C.MODEL.TRANSFORMER.after_encoder_concat = False
_C.MODEL.TRANSFORMER.fusion_backbone_encoder = False
_C.MODEL.TRANSFORMER.ENCODER = CN()
_C.MODEL.TRANSFORMER.ENCODER.n_points = 4
_C.MODEL.TRANSFORMER.ENCODER.dropout = 0.1
_C.MODEL.TRANSFORMER.ENCODER.activation = 'relu'
_C.MODEL.TRANSFORMER.ENCODER.pre_norm = True
_C.MODEL.TRANSFORMER.ENCODER.layers_rgb = 1
_C.MODEL.TRANSFORMER.ENCODER.layers_t = 1
_C.MODEL.TRANSFORMER.ENCODER.layers_fusion = 0
_C.MODEL.TRANSFORMER.ENCODER.layers_share = 0
_C.MODEL.TRANSFORMER.ENCODER.layers_4_fusion = 0
_C.MODEL.TRANSFORMER.DECODER = CN()
_C.MODEL.TRANSFORMER.DECODER.layers = 6
_C.MODEL.TRANSFORMER.DECODER.n_points = 4
_C.MODEL.TRANSFORMER.DECODER.dropout = 0.1
_C.MODEL.TRANSFORMER.DECODER.activation = 'relu'
_C.MODEL.TRANSFORMER.DECODER.pre_norm = True
_C.MODEL.TRANSFORMER.DECODER.return_intermediate = True
_C.MODEL.TRANSFORMER.DECODER.sine_embed = True
_C.MODEL.TRANSFORMER.DECODER.high_dim_query_update = False
_C.MODEL.TRANSFORMER.DECODER.fusion = True
_C.MODEL.TRANSFORMER.DECODER.fusion_concat = False
_C.MODEL.TRANSFORMER.DECODER.use_region_ca = True
_C.MODEL.TRANSFORMER.DECODER.branch_share = True
_C.MODEL.TRANSFORMER.DECODER.use_sa = True
_C.MODEL.TRANSFORMER.DECODER.key_points_det_share_dec = True
_C.MODEL.TRANSFORMER.DECODER.key_points_det_share_modality = False
_C.MODEL.TRANSFORMER.DECODER.key_points_det_share_level = True
_C.MODEL.TRANSFORMER.DECODER.KDT = CN()
_C.MODEL.TRANSFORMER.DECODER.KDT.one_layer = False
_C.MODEL.TRANSFORMER.DECODER.KDT.trick_init = False

_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.aux_loss = True
_C.MODEL.LOSS.cls_loss_coef = 1  # 1 in Deformable DETR, while 2 in DAB-DETR
_C.MODEL.LOSS.mask_loss_coef = 1
_C.MODEL.LOSS.dice_loss_coef = 1
_C.MODEL.LOSS.bbox_loss_coef = 5
_C.MODEL.LOSS.giou_loss_coef = 2
_C.MODEL.LOSS.eos_coef = 0.1
_C.MODEL.LOSS.focal_alpha = 0.25
_C.MODEL.LOSS.instance_reweight = False
_C.MODEL.LOSS.term_reweight = False
_C.MODEL.LOSS.follow_last_layer = False
_C.MODEL.LOSS.start_dynamic_weight = 0
_C.MODEL.LOSS.reweight_hard = False
_C.MODEL.LOSS.positive_alpha = [1.0, 1.0, 1.0]  # 顺序为Fusion, RGB, Thermal
_C.MODEL.LOSS.negative_alpha = [0.0, 0.0, 0.0]  # 顺序为Fusion, RGB, Thermal
_C.MODEL.LOSS.adaptive_reweight = False
_C.MODEL.LOSS.plus_three = True
_C.MODEL.LOSS.use_p = False

_C.MODEL.LOSS.gt_field_class = 'gt_labels_rgb'
_C.MODEL.LOSS.gt_field_bbox = 'gt_bboxes_rgb'

_C.MODEL.MATCHER = CN()
_C.MODEL.MATCHER.set_cost_class = 2  # Class coefficient in the matching cost
_C.MODEL.MATCHER.set_cost_bbox = 5  # L1 box coefficient in the matching cost
_C.MODEL.MATCHER.set_cost_giou = 2  # giou box coefficient in the matching cost
_C.MODEL.MATCHER.fusion_reference = False

_C.MODEL.POST_PROCESS = CN()
_C.MODEL.POST_PROCESS.num_select = 300

_C.REC = CN()
_C.REC.flag = False
_C.REC.momdality_rgb = True
_C.REC.features_loss = 'mse'
_C.REC.features_loss_coef = 1
_C.REC.freeze = True
_C.REC.checkpoint = ''

_C.DISTILL = CN()
_C.DISTILL.flag = False
_C.DISTILL.distill_modality_rgb = True
_C.DISTILL.follow_teacher = False
_C.DISTILL.distill_inter_references = False
_C.DISTILL.distill_inter_references_bbox_loss_coef = 1
_C.DISTILL.distill_inter_references_giou_loss_coef = 5
_C.DISTILL.distill_features = False
_C.DISTILL.distill_features_loss = 'L1'
_C.DISTILL.distill_features_loss_coef = 1
_C.DISTILL.distill_fusion_features = False
_C.DISTILL.rec_fusion = False
_C.DISTILL.rec_another = False
_C.DISTILL.rec_use_kernal_3 = True
_C.DISTILL.teacher_checkpoint = ''
_C.DISTILL.DECODER = CN()
_C.DISTILL.DECODER.layers = 6
_C.DISTILL.DECODER.n_points = 4
_C.DISTILL.DECODER.dropout = 0.1
_C.DISTILL.DECODER.activation = 'relu'
_C.DISTILL.DECODER.pre_norm = True
_C.DISTILL.DECODER.return_intermediate = True
_C.DISTILL.DECODER.sine_embed = True
_C.DISTILL.DECODER.high_dim_query_update = False
_C.DISTILL.DECODER.fusion = False
_C.DISTILL.DECODER.fusion_concat = False
_C.DISTILL.DECODER.use_region_ca = True
_C.DISTILL.DECODER.branch_share = True
_C.DISTILL.DECODER.use_sa = True
_C.DISTILL.DECODER.key_points_det_share_dec = True
_C.DISTILL.DECODER.key_points_det_share_modality = False
_C.DISTILL.DECODER.key_points_det_share_level = True
_C.DISTILL.DECODER.KDT = CN()
_C.DISTILL.DECODER.KDT.one_layer = False
_C.DISTILL.DECODER.KDT.trick_init = False


_C.EXPERIMENT = CN()
_C.EXPERIMENT.seed = 42
_C.EXPERIMENT.pretrain_model_path = ''
_C.EXPERIMENT.pretrain_model_path_rgb = ''
_C.EXPERIMENT.pretrain_model_path_t = ''
_C.EXPERIMENT.finetune_ignore = []
_C.EXPERIMENT.start_epoch = 0
_C.EXPERIMENT.save_results = True
_C.EXPERIMENT.save_log = True
_C.EXPERIMENT.amp = False
_C.EXPERIMENT.close_strong_augment = 1500
_C.EXPERIMENT.fit_flag = False

_C.DISTRIBUTED = CN()
_C.DISTRIBUTED.world_size = 1  # number of distributed processes
_C.DISTRIBUTED.dist_url = 'env://'  # url used to set up distributed training
_C.DISTRIBUTED.rank = 0  # number of distributed processes
_C.DISTRIBUTED.local_rank = 0  # local rank for DistributedDataParallel

_C.DATASET = CN()
_C.DATASET.format = 'kaist'
_C.DATASET.batch_size = 2
_C.DATASET.num_workers = 10
_C.DATASET.img_size = (512, 640)
_C.DATASET.KAIST = CN()
_C.DATASET.KAIST.root_dirs_train = list()
_C.DATASET.KAIST.root_dirs_test = list()
_C.DATASET.KAIST.test_gt_path = ''
_C.DATASET.KAIST.rgb_datasets_train = ['train_rgb_vbb', ]
_C.DATASET.KAIST.t_datasets_train = ['train_t_vbb', ]
_C.DATASET.KAIST.rgb_datasets_test = ['test_rgb_vbb', ]
_C.DATASET.KAIST.t_datasets_test = ['test_t_vbb', ]
_C.DATASET.KAIST.filter_mode = 2  # choices=[0, 1, 2, 3, None]
_C.DATASET.KAIST.gt_merge = False
_C.DATASET.KAIST.gt_merge_mode = 'average'
_C.DATASET.KAIST.cut_out_filter = True

_C.DATASET.TRANSFORMS = CN()
_C.DATASET.TRANSFORMS.img_size = (512, 640)
_C.DATASET.TRANSFORMS.img_resize_sizes = [(864, 1080), ]
# _C.DATASET.TRANSFORMS.img_resize_sizes = [(224, 224), ]  
_C.DATASET.TRANSFORMS.flip_ratio = 0.5
_C.DATASET.TRANSFORMS.mosaic_flag = True
_C.DATASET.TRANSFORMS.mix_up_flag = True
_C.DATASET.TRANSFORMS.random_affine_flag = True
_C.DATASET.TRANSFORMS.random_affine_before_mosaic_flag = False
_C.DATASET.TRANSFORMS.min_gt_bbox_wh = (1, 1)
_C.DATASET.TRANSFORMS.RANDAUGMENT = CN()
_C.DATASET.TRANSFORMS.RANDAUGMENT.flag = False
_C.DATASET.TRANSFORMS.RANDAUGMENT.aug_space = 'all'
_C.DATASET.TRANSFORMS.RANDAUGMENT.level = 5
_C.DATASET.TRANSFORMS.RANDAUGMENT.num = 2
_C.DATASET.TRANSFORMS.RANDAUGMENT.random = False

_C.MODEL.PROTOTYPE = CN()
_C.MODEL.PROTOTYPE.flag = False
_C.MODEL.PROTOTYPE.momentum_coef = 0.2 # 泛型更新时的参数
_C.MODEL.PROTOTYPE.loss_coef = 1  # 泛型交叉熵损失的总体权重
_C.MODEL.PROTOTYPE.reweight = True  # 是否对交叉熵损失进行reweight
_C.MODEL.PROTOTYPE.all_layers = True  # 是否对Transformer Decoder所有层进行计算泛型交叉熵损失
_C.MODEL.PROTOTYPE.alpha = 1.0 # reweight时的系数
_C.MODEL.PROTOTYPE.begin_epoch = 5 # 开始计算泛型的时间
_C.MODEL.PROTOTYPE.proportion = 0.1 # 计算泛型时使用的数据量占比
_C.MODEL.PROTOTYPE.adaptive_reweight = False


_C.TEMP = CN()
_C.TEMP.name = 'ws'


def get_cfg_defaults():
    return _C.clone()