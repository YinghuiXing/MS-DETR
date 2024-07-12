import os.path
import shutil

import cv2
import copy

from os import path
from torch.utils.data.dataset import Dataset as torchDataset
from collections import defaultdict
from tabulate import tabulate
import pprint
from pathlib import Path

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


class CVC14MultiSpectralDetection(torchDataset):
    def __init__(self, kaist_detection_rgb, kaist_detection_t, action):
        super(CVC14MultiSpectralDetection, self).__init__()
        self.kaist_detection_rgb = kaist_detection_rgb
        self.kaist_detection_t = kaist_detection_t

        assert self.kaist_detection_rgb.annotation_name == self.kaist_detection_t.annotation_name
        self.annotation_name = self.kaist_detection_rgb.annotation_name
        assert list_equal(self.kaist_detection_rgb.ids_day, self.kaist_detection_rgb.ids_day)
        assert list_equal(self.kaist_detection_rgb.ids_night, self.kaist_detection_rgb.ids_night)
        self.ids_day = self.kaist_detection_rgb.ids_day
        self.ids_night = self.kaist_detection_rgb.ids_night
        self.action = action

    def __len__(self):
        len_rgb = len(self.kaist_detection_rgb.ids)
        len_t = len(self.kaist_detection_t.ids)
        assert len_t == len_rgb
        return len_rgb

    def __getitem__(self, index):
        results_rgb = self.kaist_detection_rgb[index]
        results_t = self.kaist_detection_t[index]

        if self.action == 'train' and (results_rgb['gt_bboxes'].shape[0] == 0 or results_t['gt_bboxes'].shape[0] == 0):
            return self[(index + 1) % len(self)]

        results = self._aggregate_results_4_multimodal(results_rgb, results_t)

        return results

    def _aggregate_results_4_multimodal(self, results_rgb, results_t):
        results = copy.deepcopy(results_rgb)
        del results['img']
        results['rgb_img'] = results_rgb['img']
        results['t_img'] = results_t['img']
        results['img_fields'] = ['rgb_img', 't_img']
        results['orig_bboxes_rgb'] = results_rgb['orig_boxes']
        results['orig_bboxes_t'] = results_t['orig_boxes']

        boxes_rgb = results_rgb['gt_bboxes']
        boxes_t = results_t['gt_bboxes']
        labels_rgb = results_rgb['gt_labels']
        labels_t = results_t['gt_labels']

        assert labels_rgb.shape[0] == boxes_rgb.shape[0]
        assert labels_t.shape[0] == boxes_t.shape[0]

        min_len = min(boxes_rgb.shape[0], boxes_t.shape[0])
        if min_len == 0:
            results['gt_bboxes_rgb'] = np.zeros((0, 4))
            results['gt_bboxes_t'] = np.zeros((0, 4))
        else:
            results['gt_bboxes_rgb'] = boxes_rgb[:min_len]
            results['gt_bboxes_t'] = boxes_t[:min_len]

        results['bbox_fields'] = ['gt_bboxes_rgb', 'gt_bboxes_t']
        results['gt_labels'] = labels_rgb[:min_len]
        return results

    def draw_distinct_boxes(self):
        for i in range(len(self)):
            data = self[i]
            gt_bboxes_rgb = data['orig_bboxes_rgb']
            gt_bboxes_t = data['orig_bboxes_t']

            draw_flag = False
            unpaired_flag = False
            if len(gt_bboxes_rgb) == len(gt_bboxes_t):
                for box_rgb, box_t in zip(gt_bboxes_rgb, gt_bboxes_t):
                    residual_box = np.absolute(box_rgb - box_t)
                    if (residual_box > 20).any():
                        draw_flag = True
                        break
            else:
                draw_flag = True
                unpaired_flag = True

            if draw_flag:
                img_id_rgb = self.kaist_detection_rgb.ids[i][0]
                img_id_t = self.kaist_detection_t.ids[i][0]
                img_path_rgb = path.join(self.kaist_detection_rgb.root, img_id_rgb)
                img_path_t = path.join(self.kaist_detection_t.root, img_id_t)

                saved_base_dir = os.path.join(self.kaist_detection_rgb.root, 'unaligned_boxes')
                if unpaired_flag:
                    saved_base_dir = os.path.join(saved_base_dir, 'unpaired')
                else:
                    saved_base_dir = os.path.join(saved_base_dir, 'paired')

                saved_path_rgb = os.path.join(saved_base_dir, 'v_' + img_id_rgb)
                saved_path_t = os.path.join(saved_base_dir, 't' + img_id_t)

                Path(os.path.dirname(saved_path_rgb)).mkdir(parents=True, exist_ok=True)
                Path(os.path.dirname(saved_path_t)).mkdir(parents=True, exist_ok=True)

                boxes_decorate_rgb = list()
                boxes_decorate_t = list()
                for box in gt_bboxes_rgb:
                    current_box = list()

                    box[2] = box[2] - box[0]
                    box[3] = box[3] - box[1]

                    current_box.append(box)
                    current_box.append('red')
                    current_box.append(2.0)
                    current_box.append('')

                    boxes_decorate_rgb.append(current_box)

                for box in gt_bboxes_t:
                    current_box = list()

                    box[2] = box[2] - box[0]
                    box[3] = box[3] - box[1]

                    current_box.append(box)
                    current_box.append('red')
                    current_box.append(2.0)
                    current_box.append('')

                    boxes_decorate_t.append(current_box)

                paintBBoxes(img_path_rgb, boxes_decorate_rgb, saved_path_rgb)
                paintBBoxes(img_path_t, boxes_decorate_t, saved_path_t)


class CVC14Detection(torchDataset):
    def __init__(self, root, datasets, actionType, annotation_name, filter_mode=None):
        """
        :param root: KAIST多光谱行人检测数据集在磁盘上的根目录
        :param datasets: List，存放数据集划分文件(例如train_rgb，对应于kaist_root/datasets/train_rgb.txt，该文件每一行对应一张图像
        以及标注文件的相对路径，用逗号分隔)
        :param actionType: train, val, cam
        :param annotation_name: 标注名，取值范围为original, sanitized, paired, improve_liu, paired_all_02
        :param filter_mode: 读取标签时过滤bounding boxes的模式：
            1）None：无过滤；
            2）0：过滤掉label!='person'以及label=='person',occ=heavy的行人；
            3）1：过滤掉label not in ('person', 'cyclist')以及label == 'person', occ=heavy的行人
        """
        super(CVC14Detection, self).__init__()

        assert annotation_name == 'cvc14'

        self.root = root
        self.datasets = datasets
        self.actionType = actionType
        self.annotation_name = annotation_name
        if actionType in ('test', 'val', 'cam'):
            assert filter_mode is None
        self.filter_mode = filter_mode

        self.ids = list()
        self.ids_day = list()
        self.ids_night = list()
        self.categories = ('person',)
        self.class2index = dict(zip(self.categories, range(len(self.categories))))

        ind = 0
        for dataset in datasets:
            datasetFilePath = path.join(self.root, "datasets", "{}.txt".format(dataset))

            for line in open(datasetFilePath):
                image = line.split(',')[0].strip()
                DayOrNight = image.split('/')[0]
                if DayOrNight == 'Day':
                    self.ids_day.append(ind)
                else:
                    self.ids_night.append(ind)
                annotation = line.split(',')[1].strip()
                self.ids.append((image, annotation))
                ind += 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        results = self.pull_item(index)

        # if results['gt_bboxes'].shape[0] == 0 and self.actionType == 'train':
        #     return self[(index + 1) % len(self)]

        return results

    def pull_item(self, index):
        img_id = self.ids[index][0]

        img_path = path.join(self.root, img_id)

        img = get_img(img_path)

        anno_results = self.load_annotation(index)
        keep_ind = None
        if len(anno_results['is_filter']) > 0:
            keep_ind = np.where(~np.asarray(anno_results['is_filter']))

        results = dict()
        results['img'] = img
        results['img_fields'] = ['img', ]
        results['bbox_fields'] = ['gt_bboxes', ]
        results['ori_shape'] = img.shape
        results['img_shape'] = img.shape
        results['gt_labels'] = np.asarray(anno_results['labels'])[keep_ind] if keep_ind is not None else np.asarray(anno_results['labels'])
        results['gt_bboxes'] = np.asarray(anno_results['boxes'])[keep_ind] if len(anno_results['boxes']) > 0 else np.zeros((0, 4))
        results['orig_boxes'] = copy.deepcopy(results['gt_bboxes'])

        # for eval
        if self.actionType in ('val', 'cam', 'test', 'inference'):
            results['image_id'] = index
            results['annotation_id'] = self.ids[index][1]
            results['image_path'] = img_path

        return results

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
        if self.actionType in ('cam', 'test', 'val', 'inference'):
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
        else:
            raise RuntimeError(f'Dataset filter_mode illegal, expect 0, 1 or None, but get {self.filter_mode}')

    def load_annotation(self, index):
        annotation_id = self.ids[index][1]
        annotation_suffix = annotation_id.split('.')[1]
        assert annotation_suffix == 'txt'

        annotation_path = path.join(self.root, annotation_id)

        with open(annotation_path, 'r') as f:
            objs = f.readlines()

        boxes = list()
        obj_types = list()
        labels = list()
        isCrowd = list()
        areas = list()
        isFilter = list()
        occ_types = list()
        for obj in objs:
            obj_read = list()
            temp = obj.split('  ')
            for t in temp:
                obj_read+=t.split(' ')
            # obj = obj.split(' ')
            obj = obj_read

            # 读取数据
            obj_class = 'person'
            # import pdb
            # pdb.set_trace()
            x_centre, y_centre, width, height = float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])
            occ_type = 0

            # 过滤bbox
            filter_flag = self.is_filter('person', occ_type)
            label_id = self.class2index['person']

            boxes.append([x_centre - 0.5 * width, y_centre - 0.5 * height, x_centre + 0.5 * width, y_centre + 0.5 * height])
            obj_types.append(obj_class)
            labels.append(label_id)
            isCrowd.append(0)
            areas.append(width * height)
            isFilter.append(filter_flag)
            occ_types.append(occ_type)

        results = dict()
        results['boxes'] = boxes
        results['obj_types'] = obj_types
        results['labels'] = labels
        results['is_crowd'] = isCrowd
        results['areas'] = areas
        results['is_filter'] = isFilter
        results['occ_types'] = occ_types

        return results

    def draw_boxes(self, index):
        """
        对于指定图像，将其boxes画到图像上，将最终图像保存到/kaist_root/Images_boxes/annotation_name下
        :param index: 图像index
        """

        def get_color(cls):
            if cls == 'person':
                return 'green'
            elif cls == 'people':
                return 'yellow'
            elif cls == 'person?':
                return 'blue'
            elif cls == 'cyclist':
                return 'purple'
            elif cls == 'unpaired':
                return 'red'
            else:
                return 'pink'

        img_id = self.ids[index][0]
        img_path = path.join(self.root, img_id)

        saved_base_dir = os.path.join(self.root, 'Images_boxes', self.annotation_name)
        saved_path = os.path.join(saved_base_dir, img_id.replace('Images/', ''))

        if os.path.isfile(saved_path):
            return saved_path
        else:
            Path(os.path.dirname(saved_path)).mkdir(parents=True, exist_ok=True)

            anno_res = self.load_annotation(index)
            boxes = anno_res['boxes']
            obj_types = anno_res['obj_types']
            boxes_decorate = list()

            for box, cls in zip(boxes, obj_types):
                current_box = list()

                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]

                current_box.append(box)
                current_box.append(get_color(cls))
                current_box.append(1.0)
                current_box.append(cls)

                boxes_decorate.append(current_box)

            paintBBoxes(img_path, boxes_decorate, saved_path)
            return saved_path

    def __repr__(self):
        total_num = len(self.ids)

        dataset_info = dict()
        image_statistic = defaultdict(int)
        pedestrian_statistic = defaultdict(int)

        cyclist_ids = list()
        people_ids = list()
        person_uncertain_ids = list()
        unpaired_ids = list()

        dataset_info['root'] = self.root
        dataset_info['action_type'] = self.actionType
        dataset_info['filter_mode'] = self.filter_mode
        dataset_info['datasets'] = self.datasets

        image_statistic['image_num'] = total_num

        for ind in range(total_num):
            self.draw_boxes(ind)
            img_id = self.ids[ind][0]
            day_or_night = img_id.split('/')[0]

            anno_res = self.load_annotation(ind)
            obj_types = anno_res['obj_types']
            boxes = anno_res['boxes']
            labels = anno_res['labels']
            occ_types = anno_res['occ_types']
            is_filter = anno_res['is_filter']

            if 'cyclist' in obj_types:
                cyclist_ids.append(ind)
            elif 'people' in obj_types:
                people_ids.append(ind)
            elif 'person?' in obj_types:
                person_uncertain_ids.append(ind)
            elif 'unpaired' in obj_types:
                unpaired_ids.append(ind)

            assert len(boxes) == len(obj_types) == len(labels) == len(is_filter)

            if day_or_night == 'Day':
                image_statistic['image_num_day'] += 1
                pedestrian_statistic['pede_time_day'] += len(boxes)
            elif day_or_night == 'Night':
                image_statistic['image_num_night'] += 1
                pedestrian_statistic['pede_time_night'] += len(boxes)

            image_statistic['img_contain_{:0>2d}_objs'.format(len(boxes))] += 1

            pedestrian_statistic['pede'] += len(boxes)

            for ind_box in range(len(boxes)):
                occ_type = occ_types[ind_box]
                obj_type = obj_types[ind_box]
                height = int(boxes[ind_box][3] - boxes[ind_box][1])
                filter = is_filter[ind_box]

                if occ_type == 0:
                    pedestrian_statistic['pede_occ_none'] += 1
                elif occ_type == 1:
                    pedestrian_statistic['pede_occ_partial'] += 1
                elif occ_type == 2:
                    pedestrian_statistic['pede_occ_heavy'] += 1

                pedestrian_statistic['pede_label_' + obj_type] += 1

                if height <= 55:
                    pedestrian_statistic['pede_size_far'] += 1
                elif 55 < height <= 115:
                    pedestrian_statistic['pede_size_mid'] += 1
                elif height > 115:
                    pedestrian_statistic['pede_size_near'] += 1

                if filter:
                    pedestrian_statistic['pede_filter'] += 1

        dataset_info = [(str(k), pprint.pformat(v)) for k, v in dataset_info.items()]
        image_statistic_order = [(str(k), pprint.pformat(image_statistic[k])) for k in sorted(image_statistic.keys())]
        pedestrian_statistic_order = [(str(k), pprint.pformat(pedestrian_statistic[k])) for k in sorted(pedestrian_statistic.keys())]

        return tabulate(dataset_info + image_statistic_order + pedestrian_statistic_order, headers=['key', 'value'], tablefmt="fancy_grid")


def cvc14_split(day_split_file, night_split_file, orig_split_name, new_split_name, CVC14_root):
    with open(day_split_file, 'r') as f:
        day_files = f.readlines()
    with open(night_split_file, 'r') as f:
        night_files = f.readlines()

    def forward(files, DayOrNight, orig_split_name):
        tmp_results_visible = list()
        tmp_results_fir = list()
        for file_name in files:
            img_path_visible = os.path.join(DayOrNight, 'Visible', orig_split_name, 'FramesPos', file_name).replace('\n', '')
            img_path_fir = os.path.join(DayOrNight, 'FIR', orig_split_name, 'FramesPos', file_name).replace('\n', '')

            anno_path_visible = os.path.join(DayOrNight, 'Visible', orig_split_name, 'Annotations', file_name.split('.')[0] + '.txt')
            anno_path_fir = os.path.join(DayOrNight, 'FIR', orig_split_name, 'Annotations', file_name.split('.')[0] + '.txt')

            tmp_results_visible.append(img_path_visible + ',' + anno_path_visible + '\n')
            tmp_results_fir.append(img_path_fir + ',' + anno_path_fir + '\n')
        return tmp_results_visible, tmp_results_fir

    day_results_visible, day_results_fir = forward(day_files, 'Day', orig_split_name)
    night_results_visible, night_results_fir = forward(night_files, 'Night', orig_split_name)

    results_visible = day_results_visible + night_results_visible
    results_fir = day_results_fir + night_results_fir

    split_path_visible = os.path.join(CVC14_root, 'datasets', new_split_name + '_visible.txt')
    split_path_fir = os.path.join(CVC14_root, 'datasets', new_split_name + '_fir.txt')

    f_visible = open(split_path_visible, 'w')
    f_visible.writelines(results_visible)
    f_visible.close()

    f_fir = open(split_path_fir, 'w')
    f_fir.writelines(results_fir)
    f_fir.close()


def cvc14_test_split(CVC14_test_dir, new_split_name, CVC14_root):
    results_visible = list()
    results_fir = list()
    for root, dirs, files in os.walk(CVC14_test_dir):
        for file_name in files:
            day_or_night = file_name.split('_', 1)[0]
            file_name = file_name.split('_', 1)[1]
            time = 'Night'
            if day_or_night == 'day':
                time = 'Day'
            img_visible_path = os.path.join(time, 'Visible', 'NewTest', 'FramesPos', file_name.replace('.txt', '.tif'))
            img_fir_path = os.path.join(time, 'FIR', 'NewTest', 'FramesPos', file_name.replace('.txt', '.tif'))
            anno_visible_path = os.path.join(time, 'Visible', 'NewTest', 'Annotations', file_name)
            anno_fir_path = os.path.join(time, 'FIR', 'NewTest', 'Annotations', file_name)
            results_visible.append(img_visible_path + ',' + anno_visible_path + '\n')
            results_fir.append(img_fir_path + ',' + anno_fir_path + '\n')

    f_visible = open(os.path.join(CVC14_root, 'datasets', new_split_name + '_visible.txt'), 'w')
    f_visible.writelines(results_visible)
    f_visible.close()

    f_fir = open(os.path.join(CVC14_root, 'datasets', new_split_name + '_fir.txt'), 'w')
    f_fir.writelines(results_fir)
    f_fir.close()


def aggregete_test_files(fir_split_path, visible_split_path, CVC14_root):
    with open(fir_split_path, 'r') as f:
        firs = f.readlines()
    with open(visible_split_path, 'r') as f:
        visibles = f.readlines()

    for f in firs:
        anno = f.split(',')[1]
        day_or_night = anno.split('/')[0]
        file_name = anno.split('/')[-1].replace('\n','')
        if day_or_night == 'Day':
            day_or_night = 'day_'
        elif day_or_night == 'Night':
            day_or_night = 'night_'
        orig_file = os.path.join(CVC14_root, anno.replace('\n', ''))
        target_file = os.path.join('test_fir', day_or_night + file_name)
        shutil.copy(orig_file, target_file)

    for v in visibles:
        anno = v.split(',')[1]
        day_or_night = anno.split('/')[0]
        file_name = anno.split('/')[-1].replace('\n','')
        if day_or_night == 'Day':
            day_or_night = 'day_'
        elif day_or_night == 'Night':
            day_or_night = 'night_'
        orig_file = os.path.join(CVC14_root, anno.replace('\n', ''))
        target_file = os.path.join('test_visible', day_or_night + file_name)
        shutil.copy(orig_file, target_file)


if __name__ == '__main__-1':
    CVC14_root = '/data/wangsong/datasets/CVC-14'
    fir_split_path = os.path.join(CVC14_root, 'datasets', 'test_cvc14_fir.txt')
    visible_split_path = os.path.join(CVC14_root, 'datasets', 'test_cvc14_visible.txt')

    aggregete_test_files(fir_split_path, visible_split_path, CVC14_root)

if __name__ == '__main__0':
    CVC14_root = '/data/wangsong/datasets/CVC-14'
    # day_split_file_train = '/data/wangsong/datasets/CVC-14/Day/FIR/Train/FramesPos/files.txt'
    # night_split_file_train = '/data/wangsong/datasets/CVC-14/Night/FIR/Train/FramesPos/files.txt'
    # cvc14_split(day_split_file_train, night_split_file_train, 'Train', 'train_cvc14', CVC14_root)
    CVC14_test_dir = '/home/wangsong/exp/rgbt_ped_dect/CVC14devkit-matlab-wrapper/CVC14_annotations/test-all/annotations'
    cvc14_test_split(CVC14_test_dir, 'test_cvc14', CVC14_root)


if __name__ == '__main__':
    CVC14_root = '/data/wangsong/datasets/CVC-14'

    train_visible = CVC14Detection(CVC14_root, ['train_cvc14_visible'], 'train', 'cvc14', None)
    train_fir = CVC14Detection(CVC14_root, ['train_cvc14_fir'], 'train', 'cvc14', None)

    cvc14_multispectral_datasets = CVC14MultiSpectralDetection(train_visible, train_fir, 'train')

    cvc14_multispectral_datasets.draw_distinct_boxes()

if __name__ == '__main__2':
    visible_path = '/data/wangsong/datasets/CVC-14/Day/Visible/NewTest/FramesPos/2014_05_05_17_21_41_236000.tif'
    fir_path = '/data/wangsong/datasets/CVC-14/Day/FIR/NewTest/FramesPos/2014_05_05_17_21_41_236000.tif'
    temp_path = '/data/wangsong/datasets/KAIST/Images/set00/V000/lwir/I02233.jpg'
    img_visible = get_img(visible_path)
    img_fir = get_img(fir_path)
    img_temp = get_img(temp_path)
    print(img_visible.shape)
    print(img_fir.shape)
    print(img_temp.shape)
    print(img_visible[5, 5,:])
    print(img_visible[10, 10, :])
    print(img_visible[100, 100, :])
    print(img_visible[300, 56, :])











