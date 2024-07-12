import pprint

import cv2
import copy
import os

from os import path
import xml.etree.ElementTree as ET
from torch.utils.data.dataset import Dataset as torchDataset
from collections import defaultdict
from tabulate import tabulate
from .horizontal_boxes import HorizontalBoxes

import numpy as np

IMREAD_COLOR = 1
IMREAD_GRAYSCALE = 0
IMREAD_UNCHANGED = -1
IMREAD_IGNORE_ORIENTATION = 128


imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED,
    'color_ignore_orientation': IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    'grayscale_ignore_orientation':
        IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE
}


def get_img(img_path, to_float32=False, flag='color', channel_order='bgr'):
    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    img_np = np.frombuffer(img_bytes, np.uint8)
    flag = imread_flags[flag] if isinstance(flag, str) else flag
    img = cv2.imdecode(img_np, flag)
    if flag == IMREAD_COLOR and channel_order == 'rgb':
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

    if to_float32:
        img = img.astype(np.float32)

    return img


def list_equal(a, b):
    if len(a) != len(b):
        return False

    for a_, b_ in zip(a, b):
        if a_ != b_:
            return False
    return True


class KaistDetection(torchDataset):
    def __init__(self, roots, datasets_rgb, datasets_t, action, filter_mode=None, gt_merge=False, gt_merge_mode='average', cut_out_filter=True, just_test=False, gf_sar=False, cvc14=False):
        """
        :param root: KAIST多光谱行人检测数据集在磁盘上的根目录
        :param datasets_rgb: List，存放可见光模态的数据集划分文，例如train_rgb，对应于kaist_root/datasets/train_rgb.txt，该文件每
        一行对应一张图像以及标注文件的相对路径，用逗号分隔
        :param datasets_t: List，存放红外模态的数据集划分文件
        :param action: train, test, inference, cam
        :param filter_mode: 读取标签时过滤bounding boxes的模式：
            1）None：仅过滤掉label == unpaired的行人；
            2）0：过滤掉label!='person'以及label=='person',occ=heavy的行人；
            3）1：过滤掉label not in ('person', 'cyclist')以及label == 'person', occ=heavy的行人
            4）2：过滤掉label in ('people', 'person?')，默认项
            5）3：过滤掉label in ('people', 'person?', 'person?a')
        :param cut_out_filter: 是否在图像上裁切掉过滤掉的行人框
        :param just_test: 指示当前数据集对象仅用于测试代码正确性
        """
        super(KaistDetection, self).__init__()

        assert action in ('train', 'test', 'inference', 'cam')
        assert len(roots) == len(datasets_rgb) == len(datasets_t)
        if action != 'train' and filter_mode is not None:
            filter_mode = None

        self.roots = roots
        self.datasets_rgb = datasets_rgb
        self.datasets_t = datasets_t
        self.action = action
        self.filter_mode = filter_mode
        self.gt_merge = gt_merge
        self.gt_merge_mode = gt_merge_mode
        self.cut_out_filter = cut_out_filter
        self.just_test = just_test
        self.gf_sar = gf_sar
        self.cvc14 = cvc14

        self.ids_rgb = self.get_ids(self.datasets_rgb)
        self.ids_t = self.get_ids(self.datasets_t)

        assert self.ids_rgb or self.ids_t

        self.categories = ('person',) if not self.gf_sar else ("A220", "A330", "A320/321", "Boeing737-800", "Boeing787", "ARJ21", "other",)
        # self.categories = ('car' , 'truck' , 'bus' , 'van' , 'freight_car')
        self.class2index = dict(zip(self.categories, range(len(self.categories))))

    def __len__(self):
        if self.ids_rgb:
            return len(self.ids_rgb)

        if self.ids_t:
            return len(self.ids_t)

    def __getitem__(self, index):
        results = self.pull_item(index)

        if self.action == 'train':
            if ('gt_bboxes_rgb' in results and not results['gt_bboxes_rgb'].shape[0]) or ('gt_bboxes_t' in results and not results['gt_bboxes_t'].shape[0]):
                print('Due to no pedestrian object, skip {}-th instance'.format(index))
                if len(self.ids_rgb):
                    anno_path_rgb = self.ids_rgb[index][1]
                    print('Annotation file path of RGB modality is:{}'.format(anno_path_rgb))
                if len(self.ids_t):
                    anno_path_t = self.ids_t[index][1]
                    print('Annotation file path of Thermal modality is:{}'.format(anno_path_t))
                if not self.just_test:
                    return self[(index + 1) % len(self)]
        return results

    def get_ids(self, datasets):
        ids = list()  # 该列表中存放的是2维的元祖：(图像相对路径,标注相对路径)
        if datasets:
            ind = 0
            for root, dataset in zip(self.roots, datasets):
                dataset = path.join(root, "datasets", "{}.txt".format(dataset))

                for line in open(dataset):
                    image_relative_path = line.split(',')[0].strip()
                    annotation_relative_path = line.split(',')[1].strip()

                    image_absolute_path = os.path.join(root, image_relative_path)
                    annotation_absolute_path = os.path.join(root, annotation_relative_path)
                    ids.append((image_absolute_path, annotation_absolute_path))
                    ind += 1

        return ids

    def pull_item(self, index):
        img_absolute_path_rgb = self.ids_rgb[index][0] if self.ids_rgb else None
        anno_absolute_path_rgb = self.ids_rgb[index][1] if self.ids_rgb else None

        img_absolute_path_t = self.ids_t[index][0] if self.ids_t else None
        anno_absolute_path_t = self.ids_t[index][1] if self.ids_t else None

        img_rgb = get_img(img_absolute_path_rgb, flag='grayscale' if self.gf_sar else 'color') if img_absolute_path_rgb else None
        if anno_absolute_path_rgb:
            if self.gf_sar:
                anno_results_rgb = self.load_annotation_4_gfsar(anno_absolute_path_rgb)
            else:
                anno_results_rgb = self.load_annotation(anno_absolute_path_rgb)
        else:
            anno_results_rgb = None

        img_t = get_img(img_absolute_path_t, flag='grayscale' if self.gf_sar else 'color') if img_absolute_path_t else None
        if anno_absolute_path_t:
            if self.gf_sar:
                anno_results_t = self.load_annotation_4_gfsar(anno_absolute_path_t)
            else:
                anno_results_t = self.load_annotation(anno_absolute_path_t)
        else:
            anno_results_t = None

        #  在训练阶段，根据标注实例的属性（类型和遮挡程度），过滤掉部分标注实例
        is_filter_rgb = np.asarray(anno_results_rgb['is_filter']) if anno_results_rgb and anno_results_rgb['is_filter'] else None
        is_filter_t = np.asarray(anno_results_t['is_filter']) if anno_results_t and anno_results_t['is_filter'] else None
        img_rgb, keep_ind_rgb = self.filter_boxes(img_rgb, is_filter_rgb, anno_results_rgb['boxes']) if anno_results_rgb is not None else (None, None)
        img_t, keep_ind_t = self.filter_boxes(img_t, is_filter_t, anno_results_t['boxes']) if anno_results_t is not None else (None, None)

        results = dict()
        if img_rgb is not None and img_t is not None:
            results['ori_shape'] = img_rgb.shape
            results['rgb_img_shape'] = img_rgb.shape
            results['t_img_shape'] = img_t.shape
            results['img_shape'] = img_rgb.shape

            results['img_fields'] = ['rgb_img', 't_img']
            results['rgb_img'] = img_rgb
            results['t_img'] = img_t

            results['bbox_fields'] = ['gt_bboxes_rgb', 'gt_bboxes_t']
            results['label_fields'] = ['gt_labels_rgb', 'gt_labels_t']

            results['gt_bboxes_rgb'] = np.asarray(anno_results_rgb['boxes'])[keep_ind_rgb] if anno_results_rgb[
                'boxes'] else np.zeros((0, 4))
            results['gt_bboxes_t'] = np.asarray(anno_results_t['boxes'])[keep_ind_t] if anno_results_t[
                'boxes'] else np.zeros((0, 4))

            if self.gt_merge:
                results['gt_bboxes_rgb_before_merge'] = copy.deepcopy(results['gt_bboxes_rgb'])
                results['gt_bboxes_t_before_merge'] = copy.deepcopy(results['gt_bboxes_t'])
                merge_boxes = self.merge_gt(results['gt_bboxes_rgb'], results['gt_bboxes_t'])
                results['gt_bboxes_rgb'] = np.asarray(merge_boxes)
                results['gt_bboxes_t'] = copy.deepcopy(results['gt_bboxes_rgb'])

            results['orig_bboxes_rgb'] = copy.deepcopy(results['gt_bboxes_rgb'])
            results['orig_bboxes_t'] = copy.deepcopy(results['gt_bboxes_t'])

            results['gt_labels_rgb'] = np.asarray(anno_results_rgb['pede_labels'])[
                keep_ind_rgb] if keep_ind_rgb else np.asarray(anno_results_rgb['pede_labels'])
            results['gt_labels_t'] = np.asarray(anno_results_t['pede_labels'])[
                keep_ind_t] if keep_ind_t else np.asarray(anno_results_t['pede_labels'])

            if self.gt_merge:
                assert (results['gt_labels_rgb'] == results['gt_labels_t']).all()
                assert len(results['gt_labels_rgb']) == len(results['gt_bboxes_rgb']) == len(results['gt_labels_t']) == len(results['gt_bboxes_t'])

            if self.cvc14:
                min_len = min(results['gt_bboxes_rgb'].shape[0], results['gt_bboxes_t'].shape[0])
                if min_len == 0:
                    results['gt_bboxes_rgb'] = np.zeros((0, 4))
                    results['gt_bboxes_t'] = np.zeros((0, 4))
                    results['gt_labels_rgb'] = np.zeros(0)
                    results['gt_labels_t'] = np.zeros(0)
                else:
                    results['gt_bboxes_rgb'] = results['gt_bboxes_rgb'][:min_len]
                    results['gt_bboxes_t'] = results['gt_bboxes_t'][:min_len]
                    results['gt_labels_rgb'] = results['gt_labels_rgb'][:min_len]
                    results['gt_labels_t'] = results['gt_labels_t'][:min_len]
        elif img_rgb is not None:
            results['ori_shape'] = img_rgb.shape
            results['rgb_img_shape'] = img_rgb.shape
            results['img_shape'] = img_rgb.shape

            results['img_fields'] = ['rgb_img',]
            results['rgb_img'] = img_rgb

            results['bbox_fields'] = ['gt_bboxes_rgb', ]
            results['gt_bboxes_rgb'] = np.asarray(anno_results_rgb['boxes'])[keep_ind_rgb] if anno_results_rgb[
                'boxes'] else np.zeros((0, 4))
            results['orig_bboxes_rgb'] = copy.deepcopy(results['gt_bboxes_rgb'])

            results['label_fields'] = ['gt_labels_rgb', ]
            results['gt_labels_rgb'] = np.asarray(anno_results_rgb['pede_labels'])[
                keep_ind_rgb] if keep_ind_rgb else np.asarray(anno_results_rgb['pede_labels'])
        else:
            results['ori_shape'] = img_t.shape
            results['t_img_shape'] = img_t.shape
            results['img_shape'] = img_t.shape

            results['img_fields'] = ['t_img', ]
            results['t_img'] = img_t

            results['bbox_fields'] = ['gt_bboxes_t', ]
            results['gt_bboxes_t'] = np.asarray(anno_results_t['boxes'])[keep_ind_t] if anno_results_t[
                'boxes'] else np.zeros((0, 4))
            results['orig_bboxes_t'] = copy.deepcopy(results['gt_bboxes_t'])

            results['label_fields'] = ['gt_labels_t', ]
            results['gt_labels_t'] = np.asarray(anno_results_t['pede_labels'])[
                keep_ind_t] if keep_ind_t else np.asarray(anno_results_t['pede_labels'])

        # for eval
        if self.just_test or self.action in ('cam', 'test', 'inference'):
            results['image_ind'] = index
            results['img_absolute_path_rgb'] = img_absolute_path_rgb if len(self.ids_rgb) else None
            results['anno_absolute_path_rgb'] = anno_absolute_path_rgb if len(self.ids_rgb) else None

            results['img_absolute_path_t'] = img_absolute_path_t if len(self.ids_t) else None
            results['anno_absolute_path_t'] = anno_absolute_path_t if len(self.ids_t) else None

        return results

    def filter_boxes(self, img, is_filter, boxes):
        if is_filter is not None and len(is_filter):
            is_filter = np.asarray(is_filter)
            keep_ind = np.where(~is_filter)
            filter_ind = np.where(is_filter)
            filter_boxes = np.asarray(boxes)[filter_ind]
            if self.cut_out_filter and filter_boxes.shape[0] > 0:
                img = self.cut_out_boxes(img, filter_boxes)
        else:
            keep_ind = None
        return img, keep_ind

    def cut_out_boxes(self, img, filter_boxes):
        h, w = img.shape[:2]
        mask = np.ones((h, w, 3), dtype=img.dtype)
        for boxes in filter_boxes:
            x_0, y_0, x_1, y_1 = int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])
            mask[y_0:y_1, x_0:x_1, :] = 0

        img = img * mask
        return img

    def is_filter(self, obj_class, occ_type):
        """
        判断当前行人目标是否应该被过滤掉，具体规则如下：
            1）filter_mode is None：无过滤
            2）filter_mode == 0：过滤掉 obj_class != ’person‘ 以及 obj_class == 'person', occ_type=heavy的行人
            3）filter_mode == 1：过滤掉 obj_class not in ('person', 'cyclist') 以及 obj_class == 'person', occ_type=heavy的行人
        :param obj_class: 行人类型,可能的取值为 'person', 'people', 'person-fa', 'person?', 'cyclist'
        :param occ_type: 行人被遮挡类型，0：无遮挡，1：部分遮挡；2：严重遮挡
        :return:
        """
        if self.action in ('cam', 'test', 'val', 'inference'):
            return False
        if self.filter_mode is None:
            if obj_class == 'unpaired':
                return True
            else:
                return False
        elif self.filter_mode == 0:
            if obj_class == 'person' and occ_type != 2:
                return False
            else:
                return True
        elif self.filter_mode == 1:
            if obj_class == 'person' and occ_type != 2:
                return False
            elif obj_class == 'cyclist':
                return False
            else:
                return True
        elif self.filter_mode == 2:
            if obj_class in ('people', 'person?', 'unpaired'):
                return True
            else:
                return False
        elif self.filter_mode == 3:
            if obj_class in ('people', 'person?', 'person?a', 'unpaired'):
                return True
            else:
                return False
        else:
            raise RuntimeError(f'Dataset filter_mode illegal, expect 0, 1 or None, but get {self.filter_mode}')

    def merge_gt(self, boxes_1_p, boxes_2_p):
        boxes_1 = copy.deepcopy(boxes_1_p)
        boxes_2 = copy.deepcopy(boxes_2_p)
        assert len(boxes_1) == len(boxes_2)
        if len(boxes_1) == 0:
            return boxes_1
        else:
            res = list()
            for box_1, box_2 in zip(boxes_1, boxes_2):
                merge_box = (box_1 + box_2) // 2
                res.append(merge_box)

        return res

    def load_annotation(self, anno_absolute_path):
        if '.txt' in anno_absolute_path:
            if self. cvc14:
                return self.load_annotation_4_cvc14(anno_absolute_path)
            return self.load_annotation_4_txt(anno_absolute_path)
        elif '.xml' in anno_absolute_path:
            return self.load_annotation_4_xml(anno_absolute_path)
        else:
            raise RuntimeError

    def load_annotation_4_gfsar(self, anno_relative_path):
        anno_absolute_path = path.join(self.root, anno_relative_path)
        boxes = list()
        pede_types = list()
        pede_labels = list()
        occ_types = list()
        is_filter = list()

        target = ET.parse(anno_absolute_path).getroot()
        for obj in target.iter("object"):
            name = obj.find("possibleresult").find("name").text.strip()
            label_idx = self.class2index[name]

            points = obj.find("points")

            bndBox = []
            x_coordinates = []
            y_coordinates = []

            for point in points.iter("point"):
                coordinate = point.text.strip()
                x_coordinate = int(float(coordinate.split(",")[0]))
                y_coordinate = int(float(coordinate.split(",")[1]))
                x_coordinates.append(x_coordinate)
                y_coordinates.append(y_coordinate)

            bndBox.append(min(x_coordinates))
            bndBox.append(min(y_coordinates))
            bndBox.append(max(x_coordinates))
            bndBox.append(max(y_coordinates))
            boxes.append(bndBox)
            pede_types.append(name)
            pede_labels.append(label_idx)
            occ_types.append(0)
            is_filter.append(False)
        results = dict()
        results['boxes'] = boxes
        results['pede_types'] = pede_types
        results['pede_labels'] = pede_labels
        results['is_filter'] = is_filter
        results['occ_types'] = occ_types

        return results

    def load_annotation_4_txt(self, anno_absolute_path):
        """
        加载当前图像的单模态标注数据, 参数为该图像对应标注文件的相对地址
        标注文件格式如下:
            % bbGT version=3
            label  bbox occ_type bbox_visible ignore angle
        其中bbox, bbox_visible格式为：
            [x_upper_left, y_upper_left, w, h]
        :param anno_absolute_path: 当前图像对应标注文件的绝对地址
        :return: dict{
            'boxes': [[x, y, x, y], ..., []],
            'pede_types': ['person', ..., 'cyclist'],
            'pede_labels': [0, ..., 0],
            'is_filter': [True, ..., False]
            'occ_types': [1, ..., 2]
        }
        """
        with open(anno_absolute_path, 'r') as f:
            pedes = f.readlines()[1:]  # 标注文件第一行省略

        boxes = list()
        pede_types = list()
        pede_labels = list()
        occ_types = list()
        is_filter = list()

        for pede in pedes:
            pede = pede.split(' ')

            pede_type = pede[0]
            x_upper_left, y_upper_left, width, height = float(pede[1]), float(pede[2]), float(pede[3]), float(pede[4])
            occ_type = int(pede[5])
            label = self.class2index['person']  # 所有标注实例类别默认修改为person

            filter_flag = self.is_filter(pede_type, occ_type)  # 过滤bbox

            boxes.append([x_upper_left, y_upper_left, x_upper_left + width, y_upper_left + height])
            pede_types.append(pede_type)
            pede_labels.append(label)
            is_filter.append(filter_flag)
            occ_types.append(occ_type)

        results = dict()
        results['boxes'] = boxes
        results['pede_types'] = pede_types
        results['pede_labels'] = pede_labels
        results['is_filter'] = is_filter
        results['occ_types'] = occ_types

        return results

    def load_annotation_4_cvc14(self, anno_absolute_path):
        """
        加载当前图像的单模态标注数据, 参数为该图像对应标注文件的相对地址
        标注文件格式如下:
            label  bbox occ_type bbox_visible ignore angle
        其中bbox, bbox_visible格式为：
            [x_centre, y_centre, w, h]
        :param anno_absolute_path: 当前图像对应标注文件的绝对地址
        :return: dict{
            'boxes': [[x, y, x, y], ..., []],
            'pede_types': ['person', ..., 'cyclist'],
            'pede_labels': [0, ..., 0],
            'is_filter': [True, ..., False]
            'occ_types': [1, ..., 2]
        }
        """
        with open(anno_absolute_path, 'r') as f:
            pedes = f.readlines()

        boxes = list()
        pede_types = list()
        pede_labels = list()
        occ_types = list()
        is_filter = list()

        for pede in pedes:
            pede_read = list()

            pede = pede.split('  ') # 注意到部分标注文件中，存在着两个空格作为分割字段的符号
            for _ in pede:
                pede_read += _.split(' ')

            pede_type = 'person'
            x_centre, y_centre, width, height = float(pede_read[0]), float(pede_read[1]), float(pede_read[2]), float(pede_read[3])
            occ_type = 0
            label = self.class2index['person']  # 所有标注实例类别默认修改为person

            filter_flag = self.is_filter(pede_type, occ_type)  # 过滤bbox

            boxes.append([x_centre - 0.5 * width, y_centre - 0.5 * height, x_centre + 0.5 * width , y_centre + 0.5 * height])
            pede_types.append(pede_type)
            pede_labels.append(label)
            is_filter.append(filter_flag)
            occ_types.append(occ_type)

        results = dict()
        results['boxes'] = boxes
        results['pede_types'] = pede_types
        results['pede_labels'] = pede_labels
        results['is_filter'] = is_filter
        results['occ_types'] = occ_types

        return results

    def load_annotation_4_xml(self, anno_absolute_path):
        xmlRoot = ET.parse(anno_absolute_path).getroot()

        boxes = list()
        pede_types = list()
        pede_labels = list()
        occ_types = list()
        is_filter = list()

        for obj in xmlRoot.iter('object'):
            objClass = obj.find('name').text.lower().strip()
            pede_types.append(objClass)
            pede_labels.append(self.class2index[objClass])
            is_filter.append(False)
            occ_types.append(0)

            bbox = obj.find('bndbox')
            bndbox = list()

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in pts:
                cur_pt = float(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)

            boxes.append(bndbox)


        results = dict()
        results['boxes'] = boxes
        results['pede_types'] = pede_types
        results['pede_labels'] = pede_labels
        results['is_filter'] = is_filter
        results['occ_types'] = occ_types

        return results

    def __repr__(self):
        dataset_info = dict()
        dataset_info['root'] = self.root
        dataset_info['action'] = self.action
        dataset_info['RGB'] = False
        dataset_info['Thermal'] = False

        if len(self.ids_rgb):
            dataset_info['RGB'] = True
        if len(self.ids_t):
            dataset_info['Thermal'] = True

        if self.filter_mode is None:
            dataset_info['filter_mode'] = '过滤label == unpaired的行人'
        elif self.filter_mode == 0:
            dataset_info['filter_mode'] = '过滤label != person 以及label == person, occ = heavy的行人'
        elif self.filter_mode == 1:
            dataset_info['filter_mode'] = '过滤label not in (person, cyclist) 以及label == person, occ = heavy的行人'
        elif self.filter_mode == 2:
            dataset_info['filter_mode'] = '过滤label in (people, person?), 默认项'
        elif self.filter_mode == 3:
            dataset_info['filter_mode'] = '过滤label in (people, person?, person?a)'

        dataset_info['datasets_rgb'] = self.datasets_rgb
        dataset_info['datasets_t'] = self.datasets_t

        img_statistic_rgb, pedes_statistic_rgb, pedes_type_ids_rgb = self._get_dataset_statistic('rgb') if dataset_info['RGB'] else (None, None, None)
        img_statistic_t, pedes_statistic_t, pedes_type_ids_t = self._get_dataset_statistic('t') if dataset_info['Thermal'] else (None, None, None)

        if self.just_test:
            if pedes_type_ids_rgb is not None:
                del pedes_type_ids_rgb['person']
                print(pedes_type_ids_rgb)
            if pedes_type_ids_t is not None:
                del pedes_type_ids_t['person']
                print(pedes_type_ids_t)

        dataset_info = [(str(k), pprint.pformat(v)) for k, v in dataset_info.items()]
        if img_statistic_rgb is not None and img_statistic_t is not None:
            img_statistic_keys = sorted(list(set.union(set(img_statistic_rgb.keys()), set(img_statistic_t.keys()))))
            pedes_statistic_keys = sorted(list(set.union(set(pedes_statistic_rgb.keys()), set(pedes_statistic_t.keys()))))
        elif img_statistic_rgb is not None:
            img_statistic_keys = sorted(img_statistic_rgb.keys())
            pedes_statistic_keys = sorted(pedes_statistic_rgb.keys())
        else:
            img_statistic_keys = sorted(img_statistic_t.keys())
            pedes_statistic_keys = sorted(pedes_statistic_t.keys())

        img_statistic_info = [(str(k), pprint.pformat(img_statistic_rgb[k] if img_statistic_rgb is not None and k in img_statistic_rgb else ''), (img_statistic_t[k] if img_statistic_t is not None and k in img_statistic_t else '')) for k in img_statistic_keys]
        pedes_statistic_info = [(str(k), pprint.pformat(pedes_statistic_rgb[k] if pedes_statistic_rgb is not None and k in pedes_statistic_rgb else ''), pprint.pformat(pedes_statistic_t[k] if pedes_statistic_t is not None and k in pedes_statistic_t else '')) for k in pedes_statistic_keys]
        return tabulate(dataset_info, headers=['key', 'value'], tablefmt="fancy_grid") + '\n' + tabulate(img_statistic_info + pedes_statistic_info, headers=['key', 'value_rgb', 'value_t'], tablefmt="fancy_grid")

    def _get_dataset_statistic(self, modality):
        image_statistic = defaultdict(int)  # 统计数据集图像相关信息
        pedes_statistic = defaultdict(int)  # 统计可见光模态行人实例相关信息
        pedes_type_ids = defaultdict(list)

        assert modality in ('rgb', 't')
        total_num = len(self.ids_rgb) if modality == 'rgb' else len(self.ids_t)
        image_statistic['total_num'] += total_num
        for ind in range(total_num):
            img_relative_path = self.ids_rgb[ind][0] if modality == 'rgb' else self.ids_t[ind][0]
            anno_relative_path = self.ids_rgb[ind][1] if modality == 'rgb' else self.ids_t[ind][1]

            set_id = int(img_relative_path.split('/')[1][3:])

            anno_res = self.load_annotation(anno_relative_path)

            pede_types = anno_res['pede_types']
            boxes = anno_res['boxes']
            occ_types = anno_res['occ_types']
            is_filter = anno_res['is_filter']

            if set_id in [0, 1, 2, 6, 7, 8]:
                image_statistic['img_day'] += 1
                pedes_statistic['pedes_time_day'] += len(boxes)
            elif set_id in [3, 4, 5, 9, 10, 11]:
                image_statistic['img_night'] += 1
                pedes_statistic['pedes_time_night'] += len(boxes)

            image_statistic['img_contains_{:0>2d}_pedes'.format(len(boxes))] += 1

            pedes_statistic['pedes'] += len(boxes)

            for ind_box in range(len(boxes)):
                occ_type = occ_types[ind_box]
                pede_type = pede_types[ind_box]
                height = int(boxes[ind_box][3] - boxes[ind_box][1])
                filter = is_filter[ind_box]

                if occ_type == 0:
                    pedes_statistic['pedes_occ_none'] += 1
                elif occ_type == 1:
                    pedes_statistic['pedes_occ_partial'] += 1
                elif occ_type == 2:
                    pedes_statistic['pedes_occ_heavy'] += 1

                pedes_statistic['pedes_type_' + pede_type] += 1
                pedes_type_ids[pede_type].append(ind)

                if height <= 55:
                    pedes_statistic['pede_size_far'] += 1
                elif 55 < height <= 115:
                    pedes_statistic['pede_size_mid'] += 1
                elif height > 115:
                    pedes_statistic['pede_size_near'] += 1

                if filter:
                    pedes_statistic['pede_filter'] += 1

        return image_statistic, pedes_statistic, pedes_type_ids


def aggregate_detections(dtDir, rebuild=True, dataset_type='test', cvc14=False):
    """
    aggregate the detection files according to certain condition such as test-all, test-day and test-night

    :param dtDir: The directory of detection results, for example:/data/wangsong/results/23_9_26/exp47/test/checkpoint/fusion_branch/tenth/det
    :param rebuild: If True, rebuild the aggregated detection files even these files already existed.
    :param dataset_type: If 'test', evaluate the public test dataset that contains 2252 images;
                            If 'val', evaluate the private val dataset that provided by ws and contains 834 images
    :return:
        savedPaths: {condition: aggregated detections path}
        isEmpty: [whether the num of detection bbox under certain condition is 0]
    """
    assert dataset_type in ['test', 'val']
    savedPaths = {}
    isEmpty = []
    conditions = ['test-all', 'test-day', 'test-night']
    for cond in conditions:
        fileName = os.path.split(dtDir)[-1] + '-' + cond + '.txt' # det-test-all.txt
        fileDir = os.path.abspath(os.path.join(dtDir, '..'))  # /data/wangsong/results/23_9_26/exp47/test/checkpoint/fusion_branch/tenth
        filePath = os.path.join(fileDir, fileName) # /data/wangsong/results/23_9_26/exp47/test/checkpoint/fusion_branch/tenth/det-test-all.txt

        savedPaths[cond] = filePath

        if os.path.exists(filePath) and (not rebuild):
            continue

        # For KAIST test datasets
        # when setId is in [6, 7, 8], the images was captured during the day
        # when setId is in [9, 10, 11], the images was captured during the night
        # when setId is 6, the number of videos is 5 and so on
        # label one image every 20 frames
        if not cvc14:
            if dataset_type == 'test':
                if cond == 'test-all':
                    setIds = [6, 7, 8, 9, 10, 11]
                    skip = 20
                    videoNum = [5, 3, 3, 1, 2, 2]
                elif cond == 'test-day':
                    setIds = [6, 7, 8]
                    skip = 20
                    videoNum = [5, 3, 3]
                elif cond == 'test-night':
                    setIds = [9, 10, 11]
                    skip = 20
                    videoNum = [1, 2, 2]
            else:
                if cond == 'test-all':
                    setIds = [2, 3, 4]
                    videoNum = [2, 2, 2]
                elif cond == 'test-day':
                    setIds = [2]
                    videoNum = [2]
                elif cond == 'test-night':
                    setIds = [3, 4]
                    videoNum = [2, 2]
                skip = 2

            file = open(filePath, 'w+')
            detectionBBoxNum = 0

            num = 0
            for s in range(len(setIds)):
                for v in range(videoNum[s]):
                    for i in range(skip - 1, 99999, skip):
                        detectionFileName = 'set%02d_V%03d_I%05d.txt' % (setIds[s], v, i)  # e.g.set11_V001_I01279.txt
                        detectionFilePath = os.path.join(dtDir, detectionFileName)
                        if not os.path.exists(detectionFilePath):
                            continue
                        num += 1
                        x1, y1, x2, y2, score, attention_rgb, attention_t = [], [], [], [], [], [], []
                        detectionFile = open(detectionFilePath)

                        attention_flag = False
                        for detection in detectionFile:
                            detection = detection.strip().split(' ')

                            if len(detection) == 5: # 如果每一行长度为5
                                x1_t, y1_t, x2_t, y2_t, score_t = list(map(float, detection)) 
                            elif len(detection) == 8:
                                attention_flag = True
                                x1_t, y1_t, x2_t, y2_t, score_t, attention_rgb_t, attention_t_t = list(map(float, detection[1:]))
                            elif len(detection) == 6:
                                x1_t, y1_t, x2_t, y2_t, score_t = list(map(float, detection[1:]))
                            else:
                                raise RuntimeError

                            x1.append(x1_t)
                            x2.append(x2_t)
                            y1.append(y1_t)
                            y2.append(y2_t)
                            score.append(score_t)
                            if attention_flag:
                                attention_rgb.append(attention_rgb_t)
                                attention_t.append(attention_t_t)
                        if attention_flag:
                            for j in range(len(score)):
                                strInput = '%d,%.4f,%.4f,%.4f,%.4f,%.8f,%.8f,%.8f\n' % (num, x1[j] + 1, y1[j] + 1, x2[j] - x1[j], y2[j] - y1[j], score[j], attention_rgb[j], attention_t[j])
                                detectionBBoxNum += 1
                                file.write(strInput)
                        else:
                            for j in range(len(score)):
                                strInput = '%d,%.4f,%.4f,%.4f,%.4f,%.8f\n' % (num, x1[j] + 1, y1[j] + 1, x2[j] - x1[j], y2[j] - y1[j], score[j])
                                detectionBBoxNum += 1
                                file.write(strInput)
        else:
            file = open(filePath, 'w+')
            detectionBBoxNum = 0
            num = 0
            for dir, sub_dirs, file_names in os.walk(dtDir):
                file_names.sort()
                for file_name in file_names:
                    day_or_night = file_name.split('_')[0]
                    if cond == 'test-day' and day_or_night == 'night':
                        continue
                    if cond == 'test-night' and day_or_night == 'day':
                        continue
                    detectionFilePath = os.path.join(dir, file_name)
                    if not os.path.exists(detectionFilePath):
                        raise RuntimeError
                    num += 1
                    x1, y1, x2, y2, score = [], [], [], [], []
                    detectionFile = open(detectionFilePath)
                    for detection in detectionFile:
                        detection = detection.strip().split(' ')

                        if len(detection) == 5:
                            x1_t, y1_t, x2_t, y2_t, score_t = list(map(float, detection))
                        else:
                            x1_t, y1_t, x2_t, y2_t, score_t = list(map(float, detection[1:]))
                        x1.append(x1_t)
                        x2.append(x2_t)
                        y1.append(y1_t)
                        y2.append(y2_t)
                        score.append(score_t)
                    for j in range(len(score)):
                        strInput = '%d,%.4f,%.4f,%.4f,%.4f,%.8f\n' % (
                        num, x1[j] + 1, y1[j] + 1, x2[j] - x1[j], y2[j] - y1[j], score[j])
                        detectionBBoxNum += 1
                        file.write(strInput)

        if cond == 'test-all':
            if not cvc14:
                if dataset_type == 'test':
                    assert num == 2252, '{:d}'.format(num)
                else:
                    assert num == 834, '{:d}'.format(num)
        elif cond == 'test-day':
            if not cvc14:
                if dataset_type == 'test':
                    assert num == 1455, '{:d}'.format(num)
                else:
                    assert num == 534, '{:d}'.format(num)
        elif cond == 'test-night':
            if not cvc14:
                if dataset_type == 'test':
                    assert num == 797, '{:d}'.format(num)
                else:
                    assert num == 300, '{:d}'.format(num)

        file.close()
        if detectionBBoxNum == 0:
            isEmpty.append(True)
        else:
            isEmpty.append(False)

    return savedPaths, isEmpty









