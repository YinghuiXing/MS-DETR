import os.path

import torch
import torch.nn.functional as F
import torchvision
import pprint
from tabulate import tabulate
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Wrapper(nn.Module):
    def __init__(self, layer0, layer1, layer2, layer3, layer4):
        super().__init__()
        self.layer0 = layer0
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class VggWrapper(Wrapper):
    def __init__(self, vgg_from_torchvision):
        start_ind = 1
        vgg_features = vgg_from_torchvision.features
        layer0, start_ind = self._make_layers(vgg_features, start_ind)
        layer1, start_ind = self._make_layers(vgg_features, start_ind)
        layer2, start_ind = self._make_layers(vgg_features, start_ind)
        layer3, start_ind = self._make_layers(vgg_features, start_ind)
        layer4, start_ind = self._make_layers(vgg_features, start_ind)

        super(VggWrapper, self).__init__(layer0, layer1, layer2, layer3, layer4)

    def _make_layers(self, vgg_features, start_ind):
        layers = list()
        for idx, m in enumerate(vgg_features.modules()):
            if idx < start_ind:
                continue
            layers.append(m)
            if isinstance(m, nn.MaxPool2d):
                next_start_ind = idx + 1
                break

        return nn.Sequential(*layers), next_start_ind


class ResnetWrapper(Wrapper):
    def __init__(self, resnet_from_torchvision):
        layer0 = nn.Sequential(resnet_from_torchvision.conv1, resnet_from_torchvision.bn1, resnet_from_torchvision.relu,
                               resnet_from_torchvision.maxpool)
        super(ResnetWrapper, self).__init__(layer0, resnet_from_torchvision.layer1, resnet_from_torchvision.layer2,
                                            resnet_from_torchvision.layer3, resnet_from_torchvision.layer4)


class BackboneBase(nn.Module):
    def __init__(self, backbone_rgb, backbone_t, backbone_cfg):
        super().__init__()

        self.rgb_branch = backbone_rgb is not None
        self.t_branch = backbone_t is not None

        self._init_requires_grad(backbone_rgb, backbone_cfg.train_layers_rgb)
        self._init_requires_grad(backbone_t, backbone_cfg.train_layers_t)

        self._init_return_layers(backbone_cfg)

        self._init_feature_fusion_layers(backbone_cfg.FEATURE_FUSION)
        self._init_backbone_share_layers(backbone_rgb, backbone_t, backbone_cfg.SHARE)

        self.up_sampling_flag = backbone_cfg.UP_SAMPLING.up_sampling_flag
        self.up_sampling_stride = backbone_cfg.UP_SAMPLING.up_sampling_stride

        if self.up_sampling_flag:
            self.return_layers_strides = [self.up_sampling_stride] * len(self.return_layers_strides)

        self.body_rgb = IntermediateLayerGetter(backbone_rgb, return_layers=self.return_layers) if self.rgb_branch else None
        self.body_t = IntermediateLayerGetter(backbone_t, return_layers=self.return_layers) if self.t_branch else None

    def forward(self, tensor_list_rgb: NestedTensor, tensor_list_t: NestedTensor, targets):
        rgb_flag, t_flag, fusion_flag, share_flag = False, False, self.feature_fusion, self.share_flag

        if self.rgb_branch and tensor_list_rgb is not None:
            rgb_flag = True

        if self.t_branch and tensor_list_t is not None:
            t_flag = True

        if tensor_list_rgb is None or tensor_list_t is None:
            fusion_flag = False

        out_rgb: Dict[str, NestedTensor] = {}
        out_t: Dict[str, NestedTensor] = {}
        out_fusion: Dict[str, NestedTensor] = {}

        xs_rgb = self.body_rgb(tensor_list_rgb.tensors) if rgb_flag else None
        xs_t = self.body_t(tensor_list_t.tensors) if t_flag else None
        
        if share_flag:
            out_fusion_temp: Dict[str, NestedTensor] = {}
            for name in xs_rgb.keys():
                m = tensor_list_rgb.mask
                if int(name) == self.share_fusion_ind:
                    x_rgb = xs_rgb[name]
                    x_t = xs_t[name]
                    x_concat = torch.cat([x_rgb, x_t], dim=1)
                    x_fusion = self.share_fusion_module(x_concat)
                    mask = F.interpolate(m[None].float(), size=x_fusion.shape[-2:]).to(torch.bool)[0]
                    out_fusion_temp[name] = NestedTensor(x_fusion, mask)
                elif int(name) < self.share_fusion_ind:
                    continue
                else:
                    x_upper_layer = out_fusion_temp[str(int(name) - 1)]
                    x_fusion = getattr(self.body_rgb, 'layer' + str(name))(x_upper_layer.tensors)
                    mask = F.interpolate(m[None].float(), size=x_fusion.shape[-2:]).to(torch.bool)[0]
                    out_fusion_temp[name] = NestedTensor(x_fusion, mask)

            for name in out_fusion_temp.keys():
                if int(name) in self.share_return_layers:
                    out_fusion[name] = out_fusion_temp[name]
        else:
            if rgb_flag:
                if self.up_sampling_flag:
                    up_sampling_shape_H, up_sampling_shape_W = tensor_list_rgb.tensors.shape[-2:]
                    up_sampling_shape_H //= self.up_sampling_stride
                    up_sampling_shape_W //= self.up_sampling_stride
                    for name, x in xs_rgb.items():
                        x = F.interpolate(x, size=(up_sampling_shape_H, up_sampling_shape_W))
                        m = tensor_list_rgb.mask
                        assert m is not None
                        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                        out_rgb[name] = NestedTensor(x, mask)
                else:
                    for name, x in xs_rgb.items():
                        m = tensor_list_rgb.mask
                        assert m is not None
                        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                        out_rgb[name] = NestedTensor(x, mask)

            if t_flag:
                if self.up_sampling_flag:
                    up_sampling_shape_H, up_sampling_shape_W = tensor_list_t.tensors.shape[-2:]
                    up_sampling_shape_H //= self.up_sampling_stride
                    up_sampling_shape_W //= self.up_sampling_stride
                    for name, x in xs_t.items():
                        x = F.interpolate(x, size=(up_sampling_shape_H, up_sampling_shape_W))
                        m = tensor_list_t.mask
                        assert m is not None
                        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                        out_t[name] = NestedTensor(x, mask)
                else:
                    for name, x in xs_t.items():
                        m = tensor_list_t.mask
                        assert m is not None
                        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                        out_t[name] = NestedTensor(x, mask)


            if fusion_flag:
                for name, x_rgb in xs_rgb.items():
                    m = tensor_list_rgb.mask
                    assert m is not None
                    x_t = xs_t[name]
                    x_concat = torch.cat([x_rgb, x_t], dim=1)
                    x_fusion = getattr(self, 'layer' + name + '_fusion')(x_concat)
                    # if name == '2':
                    #     self._draw_feature_maps(x_rgb, x_t, x_fusion, name, targets)
                    mask = F.interpolate(m[None].float(), size=x_fusion.shape[-2:]).to(torch.bool)[0]
                    out_fusion[name] = NestedTensor(x_fusion, mask)

        return out_rgb, out_t, out_fusion

    def _draw_feature_maps(self, x_rgb, x_t, x_fusion, ind, targets):
        import numpy as np
        import matplotlib.pyplot as plt
        target = targets[0]
        img_absolute_path_rgb = target['img_absolute_path_rgb']
        img_absolute_path_t = target['img_absolute_path_t']
        bboxes = target['gt_bboxes_t']
        if(bboxes.shape[0] == 0):
            return

        img_name = os.path.basename(img_absolute_path_rgb)
        x_rgb = F.interpolate(x_rgb.sigmoid(), size=(471, 640))[0].cpu().numpy()
        x_rgb = np.transpose(np.mean(x_rgb, axis=0, keepdims=True), (1, 2, 0))
        x_t = F.interpolate(x_t.sigmoid(), size=(471, 640))[0].cpu().numpy()
        x_t = np.transpose(np.mean(x_t, axis=0, keepdims=True), (1, 2, 0))
        x_fusion = F.interpolate(x_fusion.sigmoid(), size=(471, 640))[0].cpu().numpy()
        x_fusion = np.transpose(np.mean(x_fusion, axis=0, keepdims=True), (1, 2, 0))

        plt.figure()
        plt.subplot()
        plt.imshow(x_rgb, cmap='spring')
        plt.axis('off')
        path = os.path.join('cvc14_7', img_name.replace('.tif', '_' + ind) + '_rgb.jpg')
        plt.savefig(path, bbox_inches='tight', pad_inches=0.0, dpi=300)

        plt.figure()
        plt.subplot()
        plt.imshow(x_t, cmap='spring')
        plt.axis('off')
        path = os.path.join('cvc14_7', img_name.replace('.tif', '_' + ind) + '_t.jpg')
        plt.savefig(path, bbox_inches='tight', pad_inches=0.0, dpi=300)

        plt.figure()
        plt.subplot()
        plt.imshow(x_fusion, cmap='spring')
        plt.axis('off')
        path = os.path.join('cvc14_7', img_name.replace('.tif', '_' + ind) + '_fusion.jpg')
        plt.savefig(path, bbox_inches='tight', pad_inches=0.0, dpi=300)

        # x_rgb_rgb = np.zeros((471, 640, 3), dtype=x_rgb.dtype)
        # x_t_rgb = np.zeros((471, 640, 3), dtype=x_t.dtype)
        # x_fusion_rgb = np.zeros((471, 640, 3), dtype=x_fusion.dtype)

        # x_rgb_rgb[:, :, 0] = x_rgb[:, :, 0]
        # x_t_rgb[:, :, 0] = x_t[:, :, 0]
        # x_fusion_rgb[:, :, 0] = x_fusion[:, :, 0]
        #
        # from visualize.visualizer import Visualizer, vis_annotation_4_three_image
        # vis_tool = Visualizer()
        #
        # vis_annotation_4_three_image(img_name.replace('.tif', '_' + ind) + '.jpg' , x_rgb_rgb, x_t_rgb, x_fusion_rgb, vis_tool, './cvc14_rgb/')

    def _init_requires_grad(self, backbone, train_layers):
        if backbone is not None:
            for name, parameter in backbone.named_parameters():
                parameter.requires_grad_(False)
                for train_layer in train_layers:
                    if 'layer' + str(train_layer) in name:
                        parameter.requires_grad_(True)

    def _init_return_layers(self, backbone_cfg):
        self.return_layers_ind = backbone_cfg.return_layers
        self.return_layers = {'layer' + str(i): str(i) for i in backbone_cfg.return_layers}
        self.return_layers_strides = [self.strides[i] for i in backbone_cfg.return_layers]
        self.return_layers_channels_rgb = [self.num_channels_rgb[i] for i in backbone_cfg.return_layers] if self.rgb_branch else None
        self.return_layers_channels_t = [self.num_channels_t[i] for i in backbone_cfg.return_layers] if self.t_branch else None

    def _init_backbone_share_layers(self, backbone_rgb, backbone_t, share_cfg):
        self.share_flag = share_cfg.share_flag
        self.num_channels_share = share_cfg.num_channels_share
        self.share_fusion_module = share_cfg.fusion_module
        self.share_fusion_ind = share_cfg.share_fusion_ind

        if self.share_flag:
            self.only_fusion = True
            self.share_fusion_module = self._make_fusion_layers(self.share_fusion_module, self.share_fusion_ind, self.num_channels_share[self.share_fusion_ind])
            self.return_layers = {'layer' + str(i): str(i) for i in [0, 1, 2, 3, 4]}
            self.share_layers_channels = [share_cfg.num_channels_share[i] for i in range(self.share_fusion_ind, 5)]
            self.return_layers_channels_share = [share_cfg.num_channels_share[i] for i in share_cfg.return_layers]
            self.share_layers = [i for i in range(self.share_fusion_ind + 1, 5)]
            self.share_return_layers = share_cfg.return_layers
            for share_ind in self.share_layers:
                layer_name = 'layer' + str(share_ind)
                setattr(backbone_rgb, layer_name, getattr(backbone_t, layer_name))

    def _init_feature_fusion_layers(self, feature_fusion_cfg):
        self.feature_fusion = feature_fusion_cfg.fusion_flag and self.rgb_branch and self.t_branch
        self.only_fusion = False
        if self.feature_fusion:
            self.only_fusion = feature_fusion_cfg.only_fusion
            self.feature_fusion_module = feature_fusion_cfg.fusion_module
            self.num_channels_fusion = feature_fusion_cfg.num_channels_fusion
            self.return_layers_channels_fusion = [feature_fusion_cfg.num_channels_fusion[i] for i in
                                                  self.return_layers_ind]

            self.layer0_fusion = self._make_fusion_layers(self.feature_fusion_module, 0, self.num_channels_fusion[0]) if 0 in self.return_layers_ind else None
            self.layer1_fusion = self._make_fusion_layers(self.feature_fusion_module, 1, self.num_channels_fusion[1]) if 1 in self.return_layers_ind else None
            self.layer2_fusion = self._make_fusion_layers(self.feature_fusion_module, 2, self.num_channels_fusion[2]) if 2 in self.return_layers_ind else None
            self.layer3_fusion = self._make_fusion_layers(self.feature_fusion_module, 3, self.num_channels_fusion[3]) if 3 in self.return_layers_ind else None
            self.layer4_fusion = self._make_fusion_layers(self.feature_fusion_module, 4, self.num_channels_fusion[4]) if 4 in self.return_layers_ind else None

    def _make_fusion_layers(self, fusion_module, layer_ind, fusion_channels):
        if fusion_module == 'conv3x3':
            fusion_module = list()
            fusion_module.append(nn.Conv2d(self.num_channels_rgb[layer_ind] + self.num_channels_t[layer_ind],
                                           fusion_channels, kernel_size=3, stride=1, padding=1, bias=False))
            fusion_module.append(nn.BatchNorm2d(fusion_channels, momentum=0.01))
            fusion_module.append(nn.ReLU(inplace=True))

            return nn.Sequential(*fusion_module)
        else:
            raise NotImplementedError

    def __repr__(self):
        backbone_info = dict()
        parameter_info = dict()
        backbone_info['RGB'] = True if self.body_rgb is not None else False
        backbone_info['Thermal'] = True if self.body_t is not None else False

        backbone_info['strides'] = self.strides
        backbone_info['num_channels_rgb'] = self.num_channels_rgb
        backbone_info['num_channels_t'] = self.num_channels_t

        backbone_info['return_layers'] = self.return_layers
        backbone_info['return_layers_strides'] = self.return_layers_strides
        backbone_info['return_layers_channels_rgb'] = self.return_layers_channels_rgb
        backbone_info['return_layers_channels_t'] = self.return_layers_channels_t

        backbone_info['feature_fusion'] = self.feature_fusion
        if self.feature_fusion:
            backbone_info['feature_fusion_module'] = self.feature_fusion_module
            backbone_info['only_fusion'] = self.only_fusion
            backbone_info['num_channels_fusion'] = self.num_channels_fusion
            backbone_info['return_layers_channels_fusion'] = self.return_layers_channels_fusion

        backbone_info['backbone_share'] = self.share_flag
        if self.share_flag:
            backbone_info['share_fusion_ind'] = self.share_fusion_ind
            backbone_info['share_layers'] = self.share_layers
            backbone_info['num_channels_share'] = self.num_channels_share
            backbone_info['share_fusion_module'] = self.share_fusion_module

        for name, parameter in self.named_parameters():
            parameter_info[name] = [parameter.numel(), parameter.requires_grad]

        backbone_info = [(str(k), pprint.pformat(v)) for k, v in backbone_info.items()]
        parameter_info = [(str(k), pprint.pformat(v[0]), pprint.pformat(v[1])) for k, v in parameter_info.items()]

        return tabulate(backbone_info, headers=['key', 'value'], tablefmt="fancy_grid") + '\n' + tabulate(
            parameter_info, headers=['name', 'parameter', 'requires_grad'], tablefmt="fancy_grid")


class Backbone(BackboneBase):
    def _init_backbone(self, name, dilation=False):
        if 'resnet' in name:
            norm_layer = FrozenBatchNorm2d
            backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation],
                                                         pretrained=is_main_process(), norm_layer=norm_layer)
            backbone = ResnetWrapper(backbone)
        elif 'vgg' in name:
            backbone = getattr(torchvision.models, name)(pretrained=is_main_process())
            backbone = VggWrapper(backbone)
        elif len(name) == 0:
            backbone = None
        else:
            raise RuntimeError(f'{name} is not supported currently!')

        return backbone

    def __init__(self, backbone_cfg):
        self.strides = backbone_cfg.strides
        self.num_channels_rgb = backbone_cfg.num_channels_rgb if backbone_cfg.backbone_rgb else None
        self.num_channels_t = backbone_cfg.num_channels_t if backbone_cfg.backbone_t else None
        self.backbone_share = backbone_cfg.backbone_share

        backbone_rgb = self._init_backbone(backbone_cfg.backbone_rgb, backbone_cfg.dilation)
        backbone_t = self._init_backbone(backbone_cfg.backbone_t, backbone_cfg.dilation) if not self.backbone_share else backbone_rgb

        super().__init__(backbone_rgb, backbone_t, backbone_cfg)

        if backbone_cfg.dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

        self.strides = backbone.return_layers_strides
        self.only_fusion = backbone.only_fusion
        self.num_channels_rgb = backbone.return_layers_channels_rgb if not self.only_fusion else None
        self.num_channels_t = backbone.return_layers_channels_t if not self.only_fusion else None
        self.backbone_share = backbone.backbone_share
        if backbone.feature_fusion:
            self.num_channels_fusion = backbone.return_layers_channels_fusion
        elif backbone.share_flag:
            self.num_channels_fusion = backbone.return_layers_channels_share
        else:
            self.num_channels_fusion = None

    def forward(self, tensor_list_rgb: NestedTensor, tensor_list_t: NestedTensor, targets):
        out_rgb: List[NestedTensor] = []
        out_t: List[NestedTensor] = []
        out_fusion: List[NestedTensor] = []

        pos_rgb = []
        pos_t = []
        pos_fusion = []

        # Firstly, extract multimodal feature maps
        xs_rgb, xs_t, xs_fusion = self[0](tensor_list_rgb, tensor_list_t, targets)

        if not self.only_fusion:
            for name, x in sorted(xs_rgb.items()):
                out_rgb.append(x)

        if not self.only_fusion:
            for name, x in sorted(xs_t.items()):
                out_t.append(x)

        if self.num_channels_fusion:
            for name, x in sorted(xs_fusion.items()):
                out_fusion.append(x)

        # Secondly, generate position encodings
        for x in out_rgb:
            pos_rgb.append(self[1](x).to(x.tensors.dtype))

        for x in out_t:
            pos_t.append(self[1](x).to(x.tensors.dtype))

        if self.num_channels_fusion:
            for x in out_fusion:
                pos_fusion.append(self[1](x).to(x.tensors.dtype))

        # Finally, predict illumination coefficients according to rgb images
        return out_rgb, out_t, out_fusion, pos_rgb, pos_t, pos_fusion


def build_backbone(backbone_cfg, position_encoding_cfg):
    backbone = Backbone(backbone_cfg)
    print(backbone)
    position_embedding = build_position_encoding(position_encoding_cfg)
    model = Joiner(backbone, position_embedding)
    return model
