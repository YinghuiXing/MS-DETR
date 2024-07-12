if __name__ == '__main__':
    import torch
    from collections import defaultdict

    _gts = defaultdict(list)
    _gts['1', '2'].append('lalal')
    print(_gts)
    


if __name__ == '__main_9_':
    import torch

    def find_integer_in_tensor(j, tensor):
        """
        检查整数 j 是否在张量 tensor 中，并返回其下标位置

        Args:
            j (int): 要检查的整数
            tensor (torch.Tensor): 输入张量

        Returns:
            (bool, torch.Tensor): 如果 j 存在于 tensor 中，则返回 True 和 j 在 tensor 中的下标位置，否则返回 False 和 None
        """
        # 使用 torch.eq 函数来检查整数是否在张量中
        result = torch.eq(tensor, j)
        indices = torch.nonzero(result, as_tuple=False)

        if indices.size(0) > 0:
            return True, indices.item()
        else:
            return False, None

# 示例用法
    tensor = torch.tensor([1, 2, 3, 4, 5])

    j = 3
    exists, indices = find_integer_in_tensor(j, tensor)
    
    print(indices.item())

    if exists:
        print(f"{j} 存在于张量中")
        print(f"{j} 在张量中的下标位置为:", indices)
    else:
        print(f"{j} 不存在于张量中")


# 测试Kaist+LLVIP数据集
if __name__ == '__main__1':
    from datasets.dataset_wrappers import MultiImageMixDataset
    from datasets.kaist import KaistDetection
    import torch
    import util.misc as utils

    rgb_train = ['train_rgb_vbb', 'LLVIP_rgb_train']
    t_train = ['train_t_vbb', 'LLVIP_t_train']

    kaist_roots = ['/data/wangsong/datasets/KAIST', '/data/wangsong/datasets/LLVIP']
    train_det = KaistDetection(kaist_roots, rgb_train, t_train, action='train', just_test=True)
    mix_train_det = MultiImageMixDataset(train_det, just_test=True)

    mix_train_det[100]


# 测试多光谱的LLVIP数据集
if __name__ == '__main__0':
    from datasets.dataset_wrappers import MultiImageMixDataset
    from datasets.kaist import KaistDetection
    import torch
    import util.misc as utils

    rgb_train = ['LLVIP_rgb_train', 'LLVIP_rgb_test']
    t_train = ['LLVIP_t_train', 'LLVIP_t_test']

    kaist_roots = ['/data/wangsong/datasets/LLVIP', '/data/wangsong/datasets/LLVIP']
    train_det = KaistDetection(kaist_roots, rgb_train, t_train, action='train', just_test=True)
    # mix_train_det = MultiImageMixDataset(train_det, just_test=True)

    dataset_length = len(train_det)
    pedestrian_num = 0
    for i in range(dataset_length):
        assert train_det[i]['gt_labels_rgb'].shape[0] == train_det[i]['gt_labels_t'].shape[0]
        pedestrian_num += train_det[i]['gt_labels_rgb'].shape[0]
    print(pedestrian_num)


if __name__ == '__main__0':
    from datasets.dataset_wrappers import MultiImageMixDataset
    from datasets.kaist import KaistDetection
    import torch
    import util.misc as utils

    rgb_train = ['train_cvc14_visible', ]
    t_train = ['train_cvc14_fir', ]

    kaist_roots = ['/data/wangsong/datasets/CVC-14', ]
    train_det = KaistDetection(kaist_roots, rgb_train, t_train, action='train', just_test=True, cvc14=True)

    dataset_length = len(train_det)
    num_not_equal = 0
    num_rgb = 0
    num_t = 0
    none_num = 0
    rgb_more_num = 0
    t_more_num = 0
    full_empty_num = 0

    for i in range(dataset_length):
        current = train_det[i]

        current_num_rgb = current['gt_labels_rgb'].shape[0]
        current_num_t = current['gt_labels_t'].shape[0]

        if current_num_rgb == 0 and current_num_t == 0:
            full_empty_num += 1

        num_rgb += current_num_rgb
        num_t += current_num_t
        if current_num_rgb != current_num_t:
            num_not_equal += 1
            if current_num_rgb > current_num_t:
                rgb_more_num += current_num_rgb - current_num_t
            else:
                t_more_num += current_num_t - current_num_rgb

    print(num_not_equal, num_rgb, num_t, rgb_more_num, t_more_num, full_empty_num)

if __name__ == '__main__0':
    from datasets.dataset_wrappers import MultiImageMixDataset
    from datasets.kaist import KaistDetection
    from visualize.visualizer import vis_annotation_4_two_image, imread, Visualizer
    import torch
    import os
    import util.misc as utils

    rgb_train = ['train_cvc14_visible', 'test_cvc14_visible']
    t_train = ['train_cvc14_fir', 'test_cvc14_fir']

    kaist_roots = ['/data/wangsong/datasets/CVC-14', '/data/wangsong/datasets/CVC-14']
    train_det = KaistDetection(kaist_roots, rgb_train, t_train, action='train', just_test=True, cvc14=True)

    dataset_length = len(train_det)

    vis_tool = Visualizer()
    for i in range(dataset_length):
        current = train_det[i]

        current_num_rgb = current['gt_labels_rgb'].shape[0]
        current_num_t = current['gt_labels_t'].shape[0]

        if current_num_rgb != current_num_t:
            saved_dir = os.path.join(kaist_roots[0], 'num_unpaired')
        elif current_num_rgb != 0:
            saved_dir = os.path.join(kaist_roots[0], 'num_paired')
        else:
            continue

        img_path_rgb = current['img_absolute_path_rgb']
        img_path_t = current['img_absolute_path_t']

        name = img_path_rgb.split('/')[-1]

        bboxes_rgb = current['gt_bboxes_rgb']
        info_rgb = [str(i) for i in range(current_num_rgb)]
        bboxes_t = current['gt_bboxes_t']
        info_t = [str(i) for i in range(current_num_t)]

        print(imread(img_path_rgb, channel_order='RGB').shape)

        vis_annotation_4_two_image(name,
                                   imread(img_path_rgb, channel_order='RGB'),
                                   imread(img_path_t, channel_order='RGB'),
                                   bboxes_rgb,
                                   None,
                                   info_rgb,
                                   bboxes_t,
                                   None,
                                   info_t,
                                   vis_tool,
                                   saved_dir
                                   )


# 测试多光谱的Kaist数据集
if __name__ == '__main__0':
    from datasets.dataset_wrappers import MultiImageMixDataset
    from datasets.kaist import KaistDetection
    import torch
    import util.misc as utils

    roots = ['/data/wangsong/datasets/KAIST', ]
    rgb_train = ['train_rgb_vbb', ]
    t_train = ['train_t_vbb', ]

    from default import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.DATASET.TRANSFORMS.random_affine_before_mosaic_flag = True
    cfg.DATASET.TRANSFORMS.random_affine_flag = False
    cfg.DATASET.TRANSFORMS.RANDAUGMENT.flag = False
    cfg.DATASET.TRANSFORMS.RANDAUGMENT.level = 5
    cfg.DATASET.TRANSFORMS.RANDAUGMENT.num = 2
    cfg.DATASET.TRANSFORMS.RANDAUGMENT.random = True
    cfg.DATASET.TRANSFORMS.RANDAUGMENT.aug_space = 'color'

    train_det = KaistDetection(roots, rgb_train, t_train, 'train', filter_mode=2, cut_out_filter=True,
                               just_test=True)
    mix_train_det = MultiImageMixDataset(train_det, mask=True, just_test=True, transforms_cfg=cfg.DATASET.TRANSFORMS)

    mix_train_det[100]

    # sampler = torch.utils.data.RandomSampler(mix_train_det)
    # batch_sampler = torch.utils.data.BatchSampler(sampler, 2, drop_last=True)
    # dataLoader = torch.utils.data.DataLoader(mix_train_det, batch_sampler=batch_sampler,
    #                                         collate_fn=utils.collate_fn, num_workers=10)

    # for a, b, c in dataLoader:
    #     print(a.tensors.shape)
    #     print(b.tensors.shape)
    #     print(c)
    #     break

# 测试多光谱的Kaist数据集(paired标注)
if __name__ == '__main__0.7':
    from datasets.dataset_wrappers import MultiImageMixDataset
    from datasets.kaist import KaistDetection
    import torch
    import util.misc as utils

    rgb_train = ['paired_train_all_rgb', ]
    t_train = ['paired_train_all_t', ]

    kaist_root = '/data/wangsong/datasets/KAIST'
    train_det = KaistDetection(kaist_root, rgb_train, t_train, 'train', filter_mode=2, cut_out_filter=True,
                               just_test=True, gt_merge=True)
    mix_train_det = MultiImageMixDataset(train_det, just_test=True)

    mix_train_det[10]
# 测试RGB模态的Kaist数据集
if __name__ == '__main__0':
    from datasets.dataset_wrappers import MultiImageMixDataset
    from datasets.kaist import KaistDetection
    import torch
    import util.misc as utils

    rgb_train = ['train_rgb_vbb', ]
    t_train = ['train_t_vbb', ]

    kaist_root = '/data/wangsong/datasets/KAIST'
    train_det = KaistDetection(kaist_root, rgb_train, None, 'train', filter_mode=2, cut_out_filter=True,
                               just_test=False)
    print(train_det)
    mix_train_det = MultiImageMixDataset(train_det, just_test=False)

    # mix_train_det[10]

    sampler = torch.utils.data.RandomSampler(mix_train_det)
    batch_sampler = torch.utils.data.BatchSampler(sampler, 2, drop_last=True)
    dataLoader = torch.utils.data.DataLoader(mix_train_det, batch_sampler=batch_sampler,
                                             collate_fn=utils.collate_fn, num_workers=10)

    for a, b, c in dataLoader:
        print(a.tensors.shape)
        print(b)
        print(c)
        break

if __name__ == '__main__1.7':
    from datasets.dataset_wrappers import MultiImageMixDataset
    from datasets.kaist import KaistDetection
    import torch
    import util.misc as utils

    rgb_train = ['gf_sar', ]

    kaist_root = '/data/wangsong/datasets/gf_sar'
    train_det = KaistDetection(kaist_root, rgb_train, None, 'train', filter_mode=2, cut_out_filter=True,
                               just_test=True, gf_sar=True)
    mix_train_det = MultiImageMixDataset(train_det, just_test=True)
    mix_train_det[196]


# 测试Thermal模态的Kaist数据集
if __name__ == '__main__2':
    from datasets.dataset_wrappers import MultiImageMixDataset, make_transforms
    from datasets.kaist import KaistDetection
    import torch
    import util.misc as utils

    rgb_train = ['train_rgb_vbb', ]
    t_train = ['train_t_vbb', ]

    kaist_root = '/data/wangsong/datasets/KAIST'
    train_det = KaistDetection(kaist_root, None, t_train, 'train', filter_mode=2, cut_out_filter=True,
                               just_test=False)
    mix_train_det = MultiImageMixDataset(train_det, just_test=False)

    # mix_train_det[10]
    sampler = torch.utils.data.RandomSampler(mix_train_det)
    batch_sampler = torch.utils.data.BatchSampler(sampler, 2, drop_last=True)
    dataLoader = torch.utils.data.DataLoader(mix_train_det, batch_sampler=batch_sampler,
                                             collate_fn=utils.collate_fn, num_workers=10)

    for a, b, c in dataLoader:
        print(a)
        print(b.tensors.shape)
        print(c)
        break

# 测试Backbone(双模态: RN50 + RN50)
if __name__ == '__main__3':
    import torch
    from default import get_cfg_defaults
    from models.dab_deformable_detr.backbone import build_backbone
    from util.misc import nested_tensor_from_tensor_list
    cfg = get_cfg_defaults()

    cfg.MODEL.BACKBONE.FEATURE_FUSION.fusion_flag = True

    backbone = build_backbone(cfg.MODEL.BACKBONE, cfg.MODEL.POSITION_ENCODING)

    img_rgb = nested_tensor_from_tensor_list([torch.randn(3, 512, 640), torch.randn(3, 512, 640)])
    img_t = nested_tensor_from_tensor_list([torch.randn(3, 512, 640), torch.randn(3, 512, 640)])

    f_rgb, f_t, f_fusion, pos_rgb, pos_t, pos_fusion = backbone(img_rgb, img_t)
    print(f_rgb[0].tensors.shape, pos_rgb[0].shape)
    print(f_rgb[1].tensors.shape, pos_rgb[1].shape)
    print(f_rgb[2].tensors.shape, pos_rgb[2].shape)

    print(f_t[0].tensors.shape, pos_t[0].shape, pos_t[0].equal(pos_rgb[0]))
    print(f_t[1].tensors.shape, pos_t[1].shape, pos_t[1].equal(pos_rgb[1]))
    print(f_t[2].tensors.shape, pos_t[2].shape, pos_t[2].equal(pos_rgb[2]))

    print(f_fusion[0].tensors.shape, pos_fusion[0].shape, pos_fusion[0].equal(pos_rgb[0]))
    print(f_fusion[1].tensors.shape, pos_fusion[1].shape, pos_fusion[1].equal(pos_rgb[1]))
    print(f_fusion[2].tensors.shape, pos_fusion[2].shape, pos_fusion[2].equal(pos_rgb[2]))

    # 输入缺省Thermal部分
    img_t = None

    f_rgb, f_t, f_fusion, pos_rgb, pos_t, pos_fusion = backbone(img_rgb, img_t)
    print(f_rgb[0].tensors.shape, pos_rgb[0].shape)
    print(f_rgb[1].tensors.shape, pos_rgb[1].shape)
    print(f_rgb[2].tensors.shape, pos_rgb[2].shape)

    print(f_t, pos_t)
    print(f_t, pos_t)
    print(f_t, pos_t)

    print(f_fusion, pos_fusion)
    print(f_fusion, pos_fusion)
    print(f_fusion, pos_fusion)

# 测试Backbone(单模态)
if __name__ == '__main__4':
    import torch
    from default import get_cfg_defaults
    from models.dab_deformable_detr.backbone import build_backbone
    from util.misc import nested_tensor_from_tensor_list
    cfg = get_cfg_defaults()

    cfg.MODEL.BACKBONE.FEATURE_FUSION.fusion_flag = True
    cfg.MODEL.BACKBONE.backbone_rgb = ''

    backbone = build_backbone(cfg.MODEL.BACKBONE, cfg.MODEL.POSITION_ENCODING)

    img_t = nested_tensor_from_tensor_list([torch.randn(3, 512, 640), torch.randn(3, 512, 640)])

    f_rgb, f_t, f_fusion, pos_rgb, pos_t, pos_fusion = backbone(None, img_t)
    print(f_rgb, pos_rgb)
    print(f_rgb, pos_rgb)
    print(f_rgb, pos_rgb)

    print(f_t[0].tensors.shape, pos_t[0].shape)
    print(f_t[1].tensors.shape, pos_t[1].shape)
    print(f_t[2].tensors.shape, pos_t[2].shape)

    print(f_fusion, pos_fusion)
    print(f_fusion, pos_fusion)
    print(f_fusion, pos_fusion)

# 测试Backbone(双模态: VGG16 + VGG16)
if __name__ == '__main__5':
    import torch
    from default import get_cfg_defaults
    from models.dab_deformable_detr.backbone import build_backbone
    from util.misc import nested_tensor_from_tensor_list
    cfg = get_cfg_defaults()

    cfg.MODEL.BACKBONE.FEATURE_FUSION.fusion_flag = True
    cfg.MODEL.BACKBONE.backbone_rgb = 'vgg16_bn'
    cfg.MODEL.BACKBONE.backbone_t = 'vgg16_bn'
    cfg.MODEL.BACKBONE.num_channels_rgb = [64, 128, 256, 512, 512]
    cfg.MODEL.BACKBONE.num_channels_t = [64, 128, 256, 512, 512]
    cfg.MODEL.BACKBONE.FEATURE_FUSION.num_channels_fusion = [64, 128, 256, 512, 512]

    backbone = build_backbone(cfg.MODEL.BACKBONE, cfg.MODEL.POSITION_ENCODING)

    img_rgb = nested_tensor_from_tensor_list([torch.randn(3, 512, 640), torch.randn(3, 512, 640)])
    img_t = nested_tensor_from_tensor_list([torch.randn(3, 512, 640), torch.randn(3, 512, 640)])

    f_rgb, f_t, f_fusion, pos_rgb, pos_t, pos_fusion = backbone(img_rgb, img_t)
    print(f_rgb[0].tensors.shape, pos_rgb[0].shape)
    print(f_rgb[1].tensors.shape, pos_rgb[1].shape)
    print(f_rgb[2].tensors.shape, pos_rgb[2].shape)

    print(f_t[0].tensors.shape, pos_t[0].shape, pos_t[0].equal(pos_rgb[0]))
    print(f_t[1].tensors.shape, pos_t[1].shape, pos_t[1].equal(pos_rgb[1]))
    print(f_t[2].tensors.shape, pos_t[2].shape, pos_t[2].equal(pos_rgb[2]))

    print(f_fusion[0].tensors.shape, pos_fusion[0].shape, pos_fusion[0].equal(pos_rgb[0]))
    print(f_fusion[1].tensors.shape, pos_fusion[1].shape, pos_fusion[1].equal(pos_rgb[1]))
    print(f_fusion[2].tensors.shape, pos_fusion[2].shape, pos_fusion[2].equal(pos_rgb[2]))


if __name__ == '__main__6':
    from default import get_cfg_defaults
    cfg = get_cfg_defaults()
    exp_config = 'exp_config/exp_2.yaml'
    cfg.merge_from_file(exp_config)
    visible_devices = ""
    for _ in cfg.CUDA.VISIBLE_DEVICES:
        visible_devices += str(_) + ","
    print(visible_devices[:-1])

if __name__ == '__main__7':
    from evaluation_script.evaluation_script import evaluate as evaluate2
    from evaluation_script.evaluation_script import draw_all
    import os
    all_path = list()
    all_path.append('/data/wangsong/results/ms-detr/exp2/test/checkpoint/fusion_branch/tenth/det-test-all.txt')

    results = [evaluate2('KAIST_annotation.json', rstFile) for rstFile in all_path]

    set_Names = ['Reasonable', 'All', 'Far', 'Medium', 'Near', 'None', 'Partial', 'Heavy']
    for ind, set_name in enumerate(set_Names):
        tmp_results = [r[set_name] for r in results]
        tmp_results = sorted(tmp_results, key=lambda x: x['all'].summarize(ind), reverse=True)
        results_img_path = os.path.join('/data/wangsong/results/ms-detr/exp2/test', 'checkpoint', 'fusion_branch', 'result_' + set_name + '.pdf')
        draw_all(tmp_results, filename=results_img_path, setting=set_name)

if __name__ == '__main__8':
    import torch.nn as nn
    k_linear = nn.Linear(256, 256, bias=True)
    print(k_linear.weight.unsqueeze(-1).unsqueeze(-1).shape)

if __name__ == '__main__9':
    import torch
    a = torch.randn(2, 3, 512, 640)
    print(a.shape[-2], type(a.shape[-2]))

if __name__ == '__main__10':
    import torch
    a = torch.randn(2, 4)
    b = torch.rand(2)
    print(a)
    print(b)
    print(a * b[..., None])

if __name__ == '__main__11':
    res = []
    for i in range(1, 2001):
        res.append(f"Images/{i}.tif,gt/{i}.xml\n")

    with open("gf_sar.txt", 'w') as f:
        f.writelines(res)


if __name__ == '__main__12':
    import torch
    pretrain_model_path = '/data/wangsong/pretrain_models/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth'
    checkpoint = torch.load(pretrain_model_path, map_location='cpu')['model']
    # for k in checkpoint.keys():
    #     print(k, checkpoint[k].shape)


    def getKeyMap(checkpoint):
        keyMap = dict()
        for k in checkpoint.keys():
            if 'level_embed' in k:
                keyMap[k.replace('level_embed', 'level_embed_rgb')] = k
                keyMap[k.replace('level_embed', 'level_embed_t')] = k
            elif 'encoder' in k:
                keyMap[k.replace('encoder', 'encoder_rgb')] = k
                keyMap[k.replace('encoder', 'encoder_t')] = k
            elif 'body' in k:
                keyMap[k.replace('body', 'body_rgb')] = k
                keyMap[k.replace('body', 'body_t')] = k
            elif 'input_proj' in k:
                keyMap[k.replace('input_proj', 'input_proj_rgb')] = k
                keyMap[k.replace('input_proj', 'input_proj_t')] = k
            elif 'decoder' not in k and 'bbox_embed' in k:
                keyMap[k.replace('bbox_embed', 'bbox_embed_rgb')] = k
                keyMap[k.replace('bbox_embed', 'bbox_embed_t')] = k
            elif 'decoder' in k or 'reference_points' in k or 'query_embed' in k:
                keyMap[k] = k
            else:
                print(k, checkpoint[k].shape)

        return keyMap


    loadMaps = {k: checkpoint[v] for k, v in getKeyMap(checkpoint).items()}
    for k in loadMaps.keys():
        if 'cross_attn.sampling_offsets.weight' in k:
            v = loadMaps[k]
            v = v.view(8, 4, 4, 2, 256)
            v = torch.cat([v, v], dim=1).view(512, 256)
            loadMaps[k] = v
        elif 'cross_attn.sampling_offsets.bias' in k:
            v = loadMaps[k]
            v = v.view(8, 4, 4, 2)
            v = torch.cat([v, v], dim=1).view(512)
            loadMaps[k] = v
        elif 'cross_attn.attention_weights.weight' in k:
            v = loadMaps[k]
            v = v.view(8, 4, 4, 1, 256)
            v = torch.cat([v, v], dim=1).view(256, 256)
            loadMaps[k] = v
        elif 'cross_attn.attention_weights.bias' in k:
            v = loadMaps[k]
            v = v.view(8, 4, 4, 1)
            v = torch.cat([v, v], dim=1).view(256)
            loadMaps[k] = v

    for k in loadMaps.keys():
        if 'cross_attn' in k:
            print(k, loadMaps[k].shape)


if __name__ == '__main__1':
    import torch
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    print(iouv)
    niou = iouv.numel()

    print(niou, torch.zeros(0, niou, dtype=torch.bool))


if __name__ == '__main__7':
    import torch
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)

            if lvl == 2:
                print(ref_x)
                print(ref_y)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)

        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    s = torch.as_tensor([[20, 20], [10, 10], [5, 5]])
    valid_ratios = torch.as_tensor([[[1, 1],[1, 1],[1, 1]], [[0.5, 0.5],[0.5, 0.5],[0.5, 0.5]]])

    r = get_reference_points(s, valid_ratios, 'cpu')
    print(r[0, 100, 2, :])
    print(r[1, 100, 2, :])

if __name__ == '__main__':
    a = [1, 2, 3, 4]
    b = a + a
    print(b)