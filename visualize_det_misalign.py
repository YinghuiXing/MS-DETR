# -*- coding: utf-8 -*-
# @Time    : 2023/7/12 16:01
# @Author  : WangSong
# @Email   : 1021707198@qq.com
# @File    : visualize_det_misalign.py
import os
from visualize.visualizer import vis_annotation_4_two_image, imread, Visualizer
import numpy as np

scene_list = ['Day', 'Night']
image_list = ['2014_05_05_17_29_23_323000.tif', '2014_05_04_23_20_07_502000.tif']
test_or_train = ['NewTest', 'Train']
root_dir = '/data/wangsong/datasets/CVC-14'
vis_tool = Visualizer()


def load_annotation_4_cvc14(anno_absolute_path):
    with open(anno_absolute_path, 'r') as f:
        pedes = f.readlines()

    boxes = list()

    for pede in pedes:
        pede_read = list()

        pede = pede.split('  ')  # 注意到部分标注文件中，存在着两个空格作为分割字段的符号
        for _ in pede:
            pede_read += _.split(' ')

        x_centre, y_centre, width, height = float(pede_read[0]), float(pede_read[1]), float(pede_read[2]), float(pede_read[3])
        boxes.append([x_centre - 0.5 * width, y_centre - 0.5 * height, x_centre + 0.5 * width, y_centre + 0.5 * height])

    return boxes


if __name__ == '__main__':
    image_path_rgb = os.path.join(root_dir, scene_list[1], 'Visible', test_or_train[1], 'FramesPos', image_list[1])
    image_path_t = os.path.join(root_dir, scene_list[1], 'FIR', test_or_train[1], 'FramesPos', image_list[1])

    gt_path_rgb = os.path.join(root_dir, scene_list[1], 'Visible', test_or_train[1], 'Annotations', image_list[1]).replace('.tif', '.txt')
    gt_path_t = os.path.join(root_dir, scene_list[1], 'FIR', test_or_train[1], 'Annotations', image_list[1].replace('tif', 'txt'))

    image_rgb = imread(image_path_rgb)
    image_t = imread(image_path_t)

    bboxes_rgb = load_annotation_4_cvc14(gt_path_rgb)
    bboxes_t = load_annotation_4_cvc14(gt_path_t)

    vis_annotation_4_two_image('cvc14_unpaired.jpg',
                               image_rgb,
                               image_t,
                               np.array(bboxes_rgb),
                               [(0, 255, 0)] * len(bboxes_rgb),
                               None,
                               np.array(bboxes_t),
                               [(0, 255, 0)] * len(bboxes_t),
                               None,
                               vis_tool,
                               'cvc14_vis'
                               )


if __name__ == '__main__1':
    image_path_rgb = os.path.join(root_dir, scene_list[0], 'Visible', test_or_train[0], 'FramesPos', image_list[0])
    image_path_t = os.path.join(root_dir, scene_list[0], 'FIR', test_or_train[0], 'FramesPos', image_list[0])

    gt_path_rgb = os.path.join(root_dir, scene_list[0], 'Visible', test_or_train[0], 'Annotations', image_list[0]).replace('.tif', '.txt')
    gt_path_t = os.path.join(root_dir, scene_list[0], 'FIR', test_or_train[0], 'Annotations', image_list[0].replace('tif', 'txt'))

    image_rgb = imread(image_path_rgb)
    image_t = imread(image_path_t)
    h, w, _ = image_rgb.shape

    bboxes_rgb = load_annotation_4_cvc14(gt_path_rgb)
    bboxes_t = load_annotation_4_cvc14(gt_path_t)

    bboxes = bboxes_rgb + bboxes_t
    max_x = 0
    max_y = 0
    for bbox in bboxes:
        max_x = max(max_x, bbox[0], bbox[2])
        max_y = max(max_y, bbox[1], bbox[3])

    max_x += 30
    max_y += 30

    max_x = int(min(max_x, w))
    max_y = int(min(max_y, h))

    max_xy = max(max_x, max_y)

    image_rgb = image_rgb[0:max_xy, 0:max_xy, :]
    image_t = image_t[0:max_xy, 0:max_xy, :]

    vis_annotation_4_two_image('cvc14_mis_align.jpg',
                               image_rgb,
                               image_t,
                               np.array(bboxes_rgb),
                               [(0, 255, 0)] * len(bboxes_rgb),
                               None,
                               np.array(bboxes_t),
                               [(0, 255, 0)] * len(bboxes_t),
                               None,
                               vis_tool,
                               'cvc14_vis'
                               )



