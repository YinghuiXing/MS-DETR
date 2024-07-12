# -*- coding: utf-8 -*-
# @Time    : 2023/7/17 19:03
# @Author  : WangSong
# @Email   : 1021707198@qq.com
# @File    : vis_motivation.py
import os, copy
import numpy as np
from visualize.visualizer import vis_annotation_4_two_image, imread, Visualizer


def visualize_det_result(rootDir, setId, videoId, imageId):
    setName = 'set%02d' % setId
    videoName = 'V%03d' % videoId
    imageName = 'I%05d.jpg' % imageId

    image_path_rgb = os.path.join(rootDir, 'Images', setName, videoName, 'visible', imageName)
    image_path_t = os.path.join(rootDir, 'Images', setName, videoName, 'lwir', imageName)

    # gt_path = os.path.join(rootDir, 'gt', 'sanitized_annotations', setName + '_' + videoName + '_' + imageName.replace('.jpg', '.txt'))
    gt_path = os.path.join(rootDir, 'gt', 'improve_annotations_liu', 'test-all', 'annotations', setName + '_' + videoName + '_' + imageName.replace('.jpg', '.txt'))


    with open(gt_path) as f:
        gts_str = f.readlines()

    del gts_str[0]
    gts = list()

    for gt_str in gts_str:
        gt_data = gt_str.strip().split()
        x0_gt, y0_gt, w_gt, h_gt = float(gt_data[1]), float(gt_data[2]), float(gt_data[3]), float(gt_data[4])

        gts.append([x0_gt, y0_gt, w_gt, h_gt])

    vis_tool = Visualizer()

    bboxes_color = list()
    des = list()
    des_rgb = ['', 'weak', 'weak', 'weak', '']
    des_t = ['weak', '', '', '', '']
    colors_rgb = [(0, 255, 0),(0, 255, 0),(0, 255, 0),(0, 255, 0)]
    colors_t = [(0, 255, 255),(0, 255, 255),(0, 255, 255),(0, 255, 255)]


    gts_copy = copy.deepcopy(gts)
    gts_copy = np.array(gts_copy)
    gts_copy[:, 2] += gts_copy[:, 0]
    gts_copy[:, 3] += gts_copy[:, 1]
    # gts_copy = np.delete(gts_copy, 2, 0)

    vis_annotation_4_two_image(setName + '_' + videoName + '_gt_' + imageName,
                                   imread(image_path_rgb, channel_order='RGB'),
                                   imread(image_path_t, channel_order='RGB'),
                                   gts_copy,
                                   colors_rgb,
                                    #None,
                                   #[str(_) for _ in range(gts_copy.shape[0])],
                               None,
                                   gts_copy,
                                   colors_t,
                               None,
                                   #[str(_) for _ in range(gts_copy.shape[0])],
                               #None,
                               vis_tool,
                                   os.path.join('vis_motivation'),
                                   concat_axis=1
                                   )


if __name__ == '__main__':
    root_dir = '/data/wangsong/datasets/KAIST'
    #visualize_det_result(root_dir, 2, 1, 187)
    # visualize_det_result(root_dir, 4, 0, 1181) # 3189 3191 3193 4 0 1181
    #visualize_det_result(root_dir, 6, 3, 2559)

    visualize_det_result(root_dir, 7, 2, 239)

    # visualize_det_result(root_dir, 9, 0, 1839)
