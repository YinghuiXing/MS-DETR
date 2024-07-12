# -*- encoding: utf-8 -*-
'''
@File    :   CSPResulut2MSDSRCNN.py    
@Contact :   shaw@mail.nwpu.edu.cn
@License :   (C)Copyright 2019-2020, XiuWeiZhangGroup-CV-NWPU

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
20-2-28 下午4:20   ShawYun      1.0         csp detection result formate -> MSDS RCNN formate
'''

import os

# from_path = '/LabData/CSP_Fusion_V3/output/valresults/KAIST/h/off2_pretrain_element_wise_subtract_abs_add_V1_NMS_0.4_score_0.1/056'
# to_path = '/LabData/CSP_Fusion_V3/output/valresults/KAIST/h/CSP2MSDSRCNN'
#
# algorithm_name = 'element_wise_subtract_abs_add_V1_NMS_0.4_score_0.1'
# model = '056'

def allmodel2MSDS(algorithm_path, algorithm_name):
    print('into')
    models = os.listdir(algorithm_path)
    to_path = '/LabData/CSP_kaist_lwir/output/valresults/KAIST/h/CSP2MSDSRCNN'
    if not os.path.exists(to_path):
        os.mkdir(to_path)

    for model in models:
        print(model)
        from_path = os.path.join(algorithm_path, model)
        CSP2MSDSRCNN(algorithm_name, model, from_path, to_path)

def CSP2MSDSRCNN(algorithm_name, model, from_path, to_path):
    algorithm_path = os.path.join(to_path, algorithm_name)
    if not os.path.exists(algorithm_path):
        os.mkdir(algorithm_path)
    model_path = os.path.join(algorithm_path, model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    dst_path = os.path.join(model_path, 'det')
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    set_dirs = os.listdir(from_path)
    for dir in set_dirs:
        dir_path = os.path.join(from_path, dir)
        v_list = os.listdir(dir_path)
        for v in v_list:
            v_path = os.path.join(dir_path, v)
            with open(v_path, 'r') as f:
                content = f.readlines()

            preLine_id = content[0].split(' ')[0]
            file_content = []
            for line in content:
                line_split = line.split(' ')

                if preLine_id == line_split[0] and line != content[-1]:
                    xmax = str(float(line_split[1]) + float(line_split[3]))
                    ymax = str(float(line_split[2]) + float(line_split[4]))
                    line_content = 'person ' + ' ' + line_split[1] + ' ' + line_split[
                        2] + ' ' + xmax + ' ' + ymax + ' ' + line_split[5]
                    file_content.append(line_content)
                else:
                    out_index = int(preLine_id.split('.')[0]) - 1
                    file_name = dir + '_' + v.replace('.txt', '') + '_' + 'I' + str("%05d" % out_index) + '.txt'
                    file_path = os.path.join(dst_path, file_name)
                    with open(file_path, 'w') as out:
                        out.writelines(file_content)

                    file_content = []
                    xmax = str(float(line_split[1]) + float(line_split[3]))
                    ymax = str(float(line_split[2]) + float(line_split[4]))
                    line_content = 'person ' + ' ' + line_split[1] + ' ' + line_split[
                        2] + ' ' + xmax + ' ' + ymax + ' ' + line_split[5]
                    file_content.append(line_content)
                # last line
                if line == content[-1]:
                    index = int(line.split('.')[0]) - 1
                    file_name = dir + '_' + v.replace('.txt', '') + '_' + 'I' + str("%05d" % index) + '.txt'
                    file_path = os.path.join(dst_path, file_name)
                    with open(file_path, 'w') as out:
                        out.writelines(file_content)

                preLine_id = line_split[0]

    # to empty object
    test_files = os.listdir("/LabData/CSP_Fusion_V3/data/KAIST/test/annotations_jingjingLiu")
    create_files = os.listdir(dst_path)
    for file in test_files:
        if file not in create_files:
            file_path = os.path.join(dst_path, file)
            with open(file_path, 'w') as emptyFin:
                emptyFin.write('')

if __name__ == '__main__':

    # path = '/LabData/CSP_Fusion_V3/output/valresults/KAIST/h/off2_pretrain_element_wise_subtract_abs_add_V1_NMS_0.4_score_0.1'
    # algorithm_name = 'element_wise_subtract_abs_add_V1_NMS_0.4_score_0.1'
    # path = '/LabData/CSP_Fusion_V3/output/valresults/KAIST/h/off2_pretrain_element_wise_subtract_abs_V1_NMS_0.4_score_0.1'
    # path = '/LabData/CSP_Fusion_V3/output/valresults/KAIST/h/off2_fusionV4_FFM_V2_all_Trainable_NMS_0.4_score_0.1'
    # path = '/LabData/CSP_Fusion_V3/output/valresults/KAIST/h/off2_pretrain_all_True_NMS_0.4_score_0.1_FWQ'
    path = '/LabData/CSP_kaist_lwir/output/valresults/KAIST/h/off2'
    algorithm_name = 'off2_lwir_NMS_0.4_score_0.1'
    allmodel2MSDS(path, algorithm_name)