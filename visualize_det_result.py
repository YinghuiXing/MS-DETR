# -*- coding: utf-8 -*-
# @Time    : 2023/7/10 10:47
# @Author  : WangSong
# @Email   : 1021707198@qq.com
# @File    : visualize_det_result.py.py
import os, copy
import numpy as np
from eval.bbGt import evalRes
from visualize.visualizer import vis_annotation_4_two_image, imread, Visualizer
import re

def visualize_det_result(rootDir, setId, videoId, imageId, index, detector, draw_gt=False, draw_dt_score_sum_list=None, draw_dt_sum_list=None,detector_ind=0):
    setName = 'set%02d' % setId
    videoName = 'V%03d' % videoId
    imageName = 'I%05d.jpg' % imageId

    image_path_rgb = os.path.join(rootDir, 'Images', setName, videoName, 'visible', imageName)
    image_path_t = os.path.join(rootDir, 'Images', setName, videoName, 'lwir', imageName)

    gt_path = os.path.join(rootDir, 'gt', 'improve_annotations_liu', 'test-all', 'annotations_KAIST_test_set', setName + '_' + videoName + '_' + imageName.replace('.jpg', '.txt'))
    dt_path = os.path.join('evaluation_script', 'state_of_arts', detector + '_result.txt')

    with open(dt_path) as f:
        dts_str = f.readlines()

    dts = list()

    for dt_str in dts_str:
        dt_data = dt_str.strip().split(',')
        dt_index = int(dt_data[0])

        if dt_index != index:
            continue
        x0_dt, y0_dt, w_gt, h_gt, score = float(dt_data[1]), float(dt_data[2]), float(dt_data[3]), float(dt_data[4]), float(dt_data[5])
        dts.append([x0_dt, y0_dt, w_gt, h_gt, score])

    with open(gt_path) as f:
        gts_str = f.readlines()

    del gts_str[0]
    gts = list()

    for gt_str in gts_str:
        gt_data = gt_str.strip().split()
        gt_label = gt_data[0]
        x0_gt, y0_gt, w_gt, h_gt = float(gt_data[1]), float(gt_data[2]), float(gt_data[3]), float(gt_data[4])
        occlusion = int(gt_data[5])

        ignore = 0

        # if gt_label != 'person':
        #     ignore = 1
        # elif h_gt < 55:
        #     ignore = 1
        # elif occlusion == 2:
        #     ignore = 1
        # elif x0_gt < 5 or x0_gt > 635 or (x0_gt + w_gt) < 5 or (x0_gt + w_gt) > 635:
        #     ignore = 1
        # elif y0_gt < 5 or y0_gt > 507 or (y0_gt + h_gt) < 5 or (y0_gt + h_gt) > 507:
        #     ignore = 1

        gts.append([x0_gt, y0_gt, w_gt, h_gt, ignore])

    vis_tool = Visualizer()

    if draw_gt:
        bboxes_color = list()
        for gt in gts:
            ignore = gt[-1]
            if ignore:
                bboxes_color.append((0, 80, 100))
            else:
                bboxes_color.append((0, 255, 255))
        # print(
        gts_copy = copy.deepcopy(gts)
        gts_copy = np.array(gts_copy)
        gts_copy = gts_copy[:, :-1]
        gts_copy[:, 2] += gts_copy[:, 0]
        gts_copy[:, 3] += gts_copy[:, 1]

        vis_annotation_4_two_image(setName + '_' + videoName + '_gt_' + imageName,
                                   imread(image_path_rgb, channel_order='RGB'),
                                   imread(image_path_t, channel_order='RGB'),
                                   gts_copy,
                                   bboxes_color,
                                   None,
                                   gts_copy,
                                   bboxes_color,
                                   None,
                                   vis_tool,
                                   os.path.join('vis_det_result_2', detector),
                                   concat_axis=0
                                   )
    if len(dts) > 0:
        dts = np.array(dts)
    else:
        dts = None

    if len(gts) > 0:
        gts = np.array(gts)
    else:
        gts = None

    gts, dts = evalRes(gts, dts)
    draw_dt_correct = list()
    draw_dt_score = list()
    draw_dt_error = list()
    for dt in dts:
        flag = dt[-1]
        if flag == 1:
            draw_dt_correct.append(list(dt[:-2]))
            draw_dt_score.append(str(round(dt[-2], 2)))
            draw_dt_score_sum_list[detector_ind] += dt[-2]
            draw_dt_sum_list[detector_ind] += 1

    for gt in gts:
        flag = gt[-1]
        if flag == 0:
            draw_dt_error.append(list(gt[:-1]))
            draw_dt_score.append('')
    
    draw_dt_correct_color = [(0, 255, 0)] * len(draw_dt_correct)
    draw_dt_error_color = [(0, 0, 255)] * len(draw_dt_error)

    draw_dt = np.array(draw_dt_correct + draw_dt_error)
    draw_dt[:, 2] += draw_dt[:, 0]
    draw_dt[:, 3] += draw_dt[:, 1]
    draw_dt_color = draw_dt_correct_color + draw_dt_error_color

    vis_annotation_4_two_image(setName + '_' + videoName + '_dt_' + imageName,
                               imread(image_path_rgb, channel_order='RGB'),
                               imread(image_path_t, channel_order='RGB'),
                               draw_dt,
                               draw_dt_color,
                               draw_dt_score,
                               draw_dt,
                               draw_dt_color,
                               draw_dt_score,
                               vis_tool,
                               os.path.join('vis_det_result_2', detector),
                               concat_axis=0
                               )


def get_index():
    # /data/wangsong/datasets/KAIST/gt/improve_annotations_liu/test-all/annotations_KAIST_test_set

    # 文件夹路径
    folder_path = '/data/wangsong/datasets/KAIST/gt/improve_annotations_liu/test-all/annotations_KAIST_test_set'

    # 正则表达式匹配文件名中的数字
    pattern = re.compile(r'set(\d+)_V(\d+)_I(\d+)')

    # 用来储存四元元组的列表
    results = []

    # 获取文件夹下的所有文件名并进行排序
    all_files = sorted(os.listdir(folder_path))

    # 遍历所有文件
    for idx, filename in enumerate(all_files):
        # 确保只处理 .txt 文件
        if filename.endswith('.txt'):
            # 使用正则表达式提取数字
            match = pattern.match(filename)
            if match:
                # 构建文件的完整路径
                file_path = os.path.join(folder_path, filename)
                # 尝试打开并读取文件
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    # 检查文件行数是否大于1
                    if len(lines) > 1:
                        # 提取 set, V, I 中的数值
                        set_num = int(match.group(1))
                        V_num = int(match.group(2))
                        I_num = int(match.group(3))
                        # 将值打包成四元元组，包括文件夹字典序下标
                        result = (set_num, V_num, I_num , idx + 1)
                        # 将四元元组添加到结果列表
                        results.append(result)
    # 打印结果列表，或者根据需要返回或保存到文件
    return results
    

if __name__ == '__main__':
    root_dir = '/data/wangsong/datasets/KAIST'
    # parameters = [(6, 3, 3499, 469), (8, 0, 1459, 1127), (9, 0, 1299, 1520), (11, 0, 799, 2114)]
    parameters = get_index()
    # print(parameters)
    # exit(0)
    # parameters = [(9, 0, 1299, 1520),]
    detectors = ['ARCNN', 'MBNet', 'MLPD', 'MS-DETR', 'GAFF' , 'MS-DETR-4-1', 'MS-DETR-4-2', 'MS-DETR-best']
    draw_dt_score_sum_list = [0] * len(detectors)
    draw_dt_sum_list = [0] * len(detectors)

    for parameter in parameters:
        for ind, detector in enumerate(detectors):
            visualize_det_result(root_dir, *parameter, detector, draw_gt=True, draw_dt_score_sum_list=draw_dt_score_sum_list, draw_dt_sum_list=draw_dt_sum_list, detector_ind=ind)
    
    for i,(score, sum) in enumerate(zip(draw_dt_score_sum_list, draw_dt_sum_list)):
        print(detectors[i], score, sum, score /sum )













