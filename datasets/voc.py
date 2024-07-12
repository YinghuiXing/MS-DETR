from os import path
from torch.utils.data.dataset import Dataset as torchDataset

import xml.etree.ElementTree as ET
import datasets.transforms as T

import numpy as np
import copy

from .dataset_wrappers import MultiImageMixDataset, MultiModalDetection

class VOCDetection(torchDataset):
    def __init__(self, root, datasets, channel_num, categories, actionType):
        super(VOCDetection, self).__init__()
        self.root = root
        self.channel_num = channel_num
        self.ids = list()
        self.class2index = dict(zip(categories, range(len(categories))))
        self.actionType = actionType

        for dataset in datasets:
            datasetFilePath = path.join(self.root, "datasets", "{}.txt".format(dataset))

            for line in open(datasetFilePath):
                image = line.split(',')[0].strip()
                annotation = line.split(',')[1].strip()
                self.ids.append((image, annotation))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        results = self.pull_item(index)

        if results['gt_bboxes'].shape[0] == 0 and self.actionType == 'train':
            return self[(index + 1) % len(self)]

        return results

    def pull_item(self, index):
        img_id = self.ids[index][0]

        set_id = int(img_id.split('/')[1][3:])
        img_path = path.join(self.root, img_id)

        img = get_img(img_path)

        boxes, labels, isCrowd, areas = self.load_annotation(index)

        results = dict()
        results['img'] = img
        results['img_fields'] = ['img', ]
        results['bbox_fields'] = ['gt_bboxes', ]
        results['ori_shape'] = img.shape
        results['img_shape'] = img.shape
        results['gt_labels'] = np.asarray(labels)
        results['gt_bboxes'] = np.asarray(boxes) if len(boxes) > 0 else np.zeros((0, 4))
        results['orig_boxes'] = copy.deepcopy(results['gt_bboxes'])
        results['illu_labels'] = [0] if set_id in [0, 1, 2] else [1]

        # for eval
        if self.actionType in ('val', 'vis', 'cam'):
            results['image_id'] = index
            results['annotation_id'] = self.ids[index][1]
            results['image_path'] = img_path

        return results

    def load_annotation(self, index):
        annotation_id = self.ids[index][1]
        annotation_path = path.join(self.root, annotation_id)

        xmlRoot = ET.parse(annotation_path).getroot()

        boxes = list()
        labels = list()
        isCrowd = list()
        areas = list()

        for obj in xmlRoot.iter('object'):
            objClass = obj.find('name').text.lower().strip()

            # KAIST test datasets have part annotation is people
            if objClass in ('people', 'person-fa', 'person?'):
                objClass = 'person'

            bbox = obj.find('bndbox')

            bndbox = list()

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in pts:
                cur_pt = float(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)

            label_id = self.class2index[objClass]

            boxes.append(bndbox)
            labels.append(label_id)
            isCrowd.append(0)
            areas.append((bndbox[2] - bndbox[0]) * (bndbox[3] - bndbox[1]))

        return boxes, labels, isCrowd, areas


def make_voc_det_transforms(image_set):
    img_scale = (512, 640)
    transforms = list()
    if image_set == 'train':
        transforms.append(T.Mosaic(img_scale=img_scale))
        transforms.append(T.RandomAffine(border=(-img_scale[0] // 2, -img_scale[1] // 2), scaling_ratio_range=(0.1, 2)))
        transforms.append(T.MixUp(img_scale=img_scale))
        transforms.append(T.RandomFlip(flip_ratio=0.5))
        transforms.append(T.Resize(img_scale=img_scale, ratio_range=(0.8, 1.2)))
        transforms.append(T.FilterAnnotations(min_gt_bbox_wh=(1, 1), keep_empty=False))
        transforms.append(T.DefaultFormatBundle())
    elif image_set in ('val', 'vis'):
        transforms.append(T.Resize(img_scale=img_scale, multiscale_mode='value'))
        transforms.append(T.DefaultFormatBundle())

    return transforms


def build(image_set, root, datasets):
    print('-------------------Build voc dataset begin-------------------')
    print('image_set: ', image_set)
    print('root: ', root)
    print('datasets: ', datasets)
    categories = ('person',)

    VOCDataset = VOCDetection(root, datasets, 3, categories, image_set)
    print('This dataset contains %d images' % len(VOCDataset))
    print('-------------------Build voc dataset end-------------------')
    return VOCDataset


def build_multimodal(image_set, root, datasets_rgb, datasets_t, backbone):
    print('-------------------Build multimodal voc dataset begin-------------------')
    print('image_set: ', image_set)
    print('root: ', root)
    print('datasets_rgb: ', datasets_rgb)
    print('datasets_t: ', datasets_t)
    print('backbone: ', backbone)
    categories = ('person',)
    VOCDataset_RGB = VOCDetection(root, datasets_rgb, 3, categories, image_set)
    VOCDataset_T = VOCDetection(root, datasets_t, 3, categories, image_set)

    MultiModalDataset = MultiModalDetection(VOCDataset_RGB, VOCDataset_T)
    AugDataset = MultiImageMixDataset(MultiModalDataset, make_voc_det_transforms(image_set), image_set)

    print('This dataset contains %d images' % len(AugDataset))
    print('-------------------Build multimodal voc dataset end-------------------')
    return AugDataset


if __name__ == '__main__':
    a = 'train'
    b = '/data/wangsong/datasets/KAIST'
    c = ['train_rgb', ]
    d = ['train_t', ]
    e = 'Resnet50'

    f = build_multimodal(a, b, c, d, e)

    print(len(f))
    f.update_skip_type_keys(["Mosaic", "RandomAffine", "MixUp"])
    f[101]
