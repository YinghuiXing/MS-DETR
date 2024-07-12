# # -*- coding: utf-8 -*-
# # @Time    : 2024/4/27 20:52
# # @Author  : YangShuo
# # @Email   : 
# # @File    : visualize_det_result_cvc14.py.py
# import os, copy
# import numpy as np
# from eval.bbGt import evalRes
# from visualize.visualizer import vis_annotation_4_two_image, imread, Visualizer
# import re



# # def vis_annotation_4_two_image(name: str,
# #                        image_1: np.ndarray,
# #                        image_2: np.ndarray,
# #                        bboxes_1: np.ndarray,
# #                        bboxes_color_1,
# #                        info_1: Optional[Union[str, List[str]]],
# #                        bboxes_2: np.ndarray,
# #                        bboxes_color_2,
# #                        info_2: Optional[Union[str, List[str]]],
# #                        visualizer: Visualizer,
# #                        save_dir: str,
# #                        concat_axis: int = 0,
# #                        ) -> None:
# #     image_1 = image_1.clip(0, 255).astype(np.uint8)
# #     image_2 = image_2.clip(0, 255).astype(np.uint8)

# #     image_1 = draw_bboxes_4_single_image(image_1, bboxes_1, bboxes_color_1, info_1, visualizer=visualizer)
# #     image_2 = draw_bboxes_4_single_image(image_2, bboxes_2, bboxes_color_2, info_2, visualizer=visualizer)

# #     img = np.concatenate((image_1, image_2), axis=concat_axis)  # concat_axis==0表示垂直，==1表示水平

# #     img_path = os.path.join(save_dir, name)
# #     imwrite(img, img_path)


# def visualize_det_result(rootDir,  index, detector, draw_gt=False, draw_dt_score_sum_list=None, draw_dt_sum_list=None,detector_ind=0):


#     image_path_rgb = os.path.join(rootDir, 'Images', setName, videoName, 'visible', imageName)
#     image_path_t = os.path.join(rootDir, 'Images', setName, videoName, 'lwir', imageName)

#     # 两个模态的gt txt路径
#     rgb_gt_path = os.path.join(rootDir, 'gt', 'improve_annotations_liu', 'test-all', 'annotations_KAIST_test_set', setName + '_' + videoName + '_' + imageName.replace('.jpg', '.txt'))
#     t_gt_path = None
#     # cvc14的检测结果txt
#     dt_path = os.path.join('evaluation_script', 'state_of_arts', detector + '_result.txt')

#     with open(dt_path) as f:
#         dts_str = f.readlines()

#     dts = list()

#     for dt_str in dts_str:
#         dt_data = dt_str.strip().split(',')
#         dt_index = int(dt_data[0])

#         if dt_index != index:
#             continue
#         x0_dt, y0_dt, w_gt, h_gt, score = float(dt_data[1]), float(dt_data[2]), float(dt_data[3]), float(dt_data[4]), float(dt_data[5])
#         dts.append([x0_dt, y0_dt, w_gt, h_gt, score])

#     with open(gt_path) as f:
#         gts_str = f.readlines()

#     del gts_str[0]
#     gts = list()

#     for gt_str in gts_str:
#         gt_data = gt_str.strip().split()
#         gt_label = gt_data[0]
#         x0_gt, y0_gt, w_gt, h_gt = float(gt_data[1]), float(gt_data[2]), float(gt_data[3]), float(gt_data[4])
#         occlusion = int(gt_data[5])

#         ignore = 0
#         gts.append([x0_gt, y0_gt, w_gt, h_gt, ignore])

#     vis_tool = Visualizer()

#     if draw_gt:
#         bboxes_color = list()
#         for gt in gts:
#             ignore = gt[-1]
#             if ignore:
#                 bboxes_color.append((0, 80, 100))
#             else:
#                 bboxes_color.append((0, 255, 255))
#         # print(
#         gts_copy = copy.deepcopy(gts)
#         gts_copy = np.array(gts_copy)
#         gts_copy = gts_copy[:, :-1]
#         gts_copy[:, 2] += gts_copy[:, 0]
#         gts_copy[:, 3] += gts_copy[:, 1]

#         vis_annotation_4_two_image(setName + '_' + videoName + '_gt_' + imageName,
#                                    imread(image_path_rgb, channel_order='RGB'),
#                                    imread(image_path_t, channel_order='RGB'),
#                                    gts_copy,
#                                    bboxes_color,
#                                    None,
#                                    gts_copy,
#                                    bboxes_color,
#                                    None,
#                                    vis_tool,
#                                    os.path.join('vis_det_result_2', detector),
#                                    concat_axis=0
#                                    )
    
    
#     if len(dts) > 0:
#         dts = np.array(dts)
#     else:
#         dts = None

#     if len(gts) > 0:
#         gts = np.array(gts)
#     else:
#         gts = None

#     gts, dts = evalRes(gts, dts)
#     draw_dt_correct = list()
#     draw_dt_score = list()
#     draw_dt_error = list()
#     for dt in dts:
#         flag = dt[-1]
#         if flag == 1:
#             draw_dt_correct.append(list(dt[:-2]))
#             draw_dt_score.append(str(round(dt[-2], 2)))
#             draw_dt_score_sum_list[detector_ind] += dt[-2]
#             draw_dt_sum_list[detector_ind] += 1

#     for gt in gts:
#         flag = gt[-1]
#         if flag == 0:
#             draw_dt_error.append(list(gt[:-1]))
#             draw_dt_score.append('')
    
#     draw_dt_correct_color = [(0, 255, 0)] * len(draw_dt_correct)
#     draw_dt_error_color = [(0, 0, 255)] * len(draw_dt_error)

#     draw_dt = np.array(draw_dt_correct + draw_dt_error)
#     draw_dt[:, 2] += draw_dt[:, 0]
#     draw_dt[:, 3] += draw_dt[:, 1]
#     draw_dt_color = draw_dt_correct_color + draw_dt_error_color

#     vis_annotation_4_two_image(setName + '_' + videoName + '_dt_' + imageName,
#                                imread(image_path_rgb, channel_order='RGB'),
#                                imread(image_path_t, channel_order='RGB'),
#                                draw_dt,
#                                draw_dt_color,
#                                draw_dt_score,
#                                draw_dt,
#                                draw_dt_color,
#                                draw_dt_score,
#                                vis_tool,
#                                os.path.join('vis_det_result_2', detector),
#                                concat_axis=0
#                                )


# # def get_index():
# #     # /data/wangsong/datasets/KAIST/gt/improve_annotations_liu/test-all/annotations_KAIST_test_set

# #     # 文件夹路径
# #     folder_path = '/data/wangsong/datasets/KAIST/gt/improve_annotations_liu/test-all/annotations_KAIST_test_set'

# #     # 正则表达式匹配文件名中的数字
# #     pattern = re.compile(r'set(\d+)_V(\d+)_I(\d+)')

# #     # 用来储存四元元组的列表
# #     results = []

# #     # 获取文件夹下的所有文件名并进行排序
# #     all_files = sorted(os.listdir(folder_path))

# #     # 遍历所有文件
# #     for idx, filename in enumerate(all_files):
# #         # 确保只处理 .txt 文件
# #         if filename.endswith('.txt'):
# #             # 使用正则表达式提取数字
# #             match = pattern.match(filename)
# #             if match:
# #                 # 构建文件的完整路径
# #                 file_path = os.path.join(folder_path, filename)
# #                 # 尝试打开并读取文件
# #                 with open(file_path, 'r') as file:
# #                     lines = file.readlines()
# #                     # 检查文件行数是否大于1
# #                     if len(lines) > 1:
# #                         # 提取 set, V, I 中的数值
# #                         set_num = int(match.group(1))
# #                         V_num = int(match.group(2))
# #                         I_num = int(match.group(3))
# #                         # 将值打包成四元元组，包括文件夹字典序下标
# #                         result = (set_num, V_num, I_num , idx + 1)
# #                         # 将四元元组添加到结果列表
# #                         results.append(result)
# #     # 打印结果列表，或者根据需要返回或保存到文件
# #     return results
    

# if __name__ == '__main__':
#     pass
#     # root_dir = '/data/wangsong/datasets/KAIST'
#     # parameters = [(6, 3, 3499, 469), (8, 0, 1459, 1127), (9, 0, 1299, 1520), (11, 0, 799, 2114)]
#     # parameters = get_index()
#     # print(parameters)
#     # exit(0)
#     # parameters = [(9, 0, 1299, 1520),]
#     # detectors = ['ARCNN', 'MBNet', 'MLPD', 'MS-DETR', 'GAFF' , 'MS-DETR-4-1', 'MS-DETR-4-2', 'MS-DETR-best']
#     # draw_dt_score_sum_list = [0] * len(detectors)
#     # draw_dt_sum_list = [0] * len(detectors)

#     # for parameter in parameters:
#     #     for ind, detector in enumerate(detectors):
#     #         visualize_det_result(root_dir, *parameter, detector, draw_gt=True, draw_dt_score_sum_list=draw_dt_score_sum_list, draw_dt_sum_list=draw_dt_sum_list, detector_ind=ind)
    
#     # for i,(score, sum) in enumerate(zip(draw_dt_score_sum_list, draw_dt_sum_list)):
#     #     print(detectors[i], score, sum, score /sum )


# rgb_txt_root = None 
# t_txt_root = None
# rgb_img_root = None 
# t_img_root = None 



    

# import os
# from PIL import Image, ImageDraw

# # 定义文件夹路径
# visible_img_folder = '/data/wangsong/datasets/CVC-14/Day/Visible/NewTest/FramesPos'
# infrared_img_folder = '/data/wangsong/datasets/CVC-14/Day/FIR/NewTest/FramesPos'
# visible_ann_folder = '/data/wangsong/datasets/CVC-14/Day/Visible/NewTest/Annotations'
# infrared_ann_folder = '/data/wangsong/datasets/CVC-14/Day/FIR/NewTest/Annotations'
# result_folder = './cvc14_result/'

# # 确保结果文件夹存在
# os.makedirs(result_folder, exist_ok=True)

# # 读取标注并绘制到图片上的函数
# def draw_annotations(image_path, annotation_path):
#     # 打开图片和标注文件
#     with Image.open(image_path) as img:
#         draw = ImageDraw.Draw(img)
#         with open(annotation_path, 'r') as file:
#             for line in file:
#                 # 解析标注的x, y, w, h
#                 x, y, w, h = map(int, line.split()[:4])
#                 # 绘制矩形框
#                 draw.rectangle(((x, y), (x + w, y + h)), outline='red', width=2)
#         return img

# # 遍历可见光图片文件夹
# for filename in os.listdir(visible_img_folder):
#     if filename.endswith('.tif'):
#         # 构建对应的文件路径
#         visible_img_path = os.path.join(visible_img_folder, filename)
#         infrared_img_path = os.path.join(infrared_img_folder, filename)
#         visible_ann_path = os.path.join(visible_ann_folder, os.path.splitext(filename)[0] + '.txt')
#         infrared_ann_path = os.path.join(infrared_ann_folder, os.path.splitext(filename)[0] + '.txt')
        
#         # 绘制标注
#         visible_img_annotated = draw_annotations(visible_img_path, visible_ann_path)
#         infrared_img_annotated = draw_annotations(infrared_img_path, infrared_ann_path)
        
#         # 拼接图片
#         combined_img = Image.new('RGB', (visible_img_annotated.width, visible_img_annotated.height + infrared_img_annotated.height))
#         combined_img.paste(visible_img_annotated, (0, 0))
#         combined_img.paste(infrared_img_annotated, (0, visible_img_annotated.height))
        
#         # 保存结果图片
#         result_img_path = os.path.join(result_folder, filename)
#         combined_img.save(result_img_path)

# print("Annotation visualization completed.")







###################################################################################################################################################################


# 现在已经有模型的检测出来的文件了，都是txt格式的，每一个txt对应一个图片的检测结果，每一行代表着一个实例，比如某个文件的其中一行“person 560.1946411132812 82.12245178222656 574.8495483398438 128.32008361816406 0.956481397151947”
# 形式为“类别 x1 y1 x2 y2 score”。
# txt的文件名称命名格式为"day_图片文件名"，代表这个图片在Day文件夹下，对应着Day/Visible/NewTest/FramesPos文件夹和Day/FIR/NewTest/FramesPos文件夹各有对应模态的图片，我希望你把txt的标注绘制在两个模态的图片上，并将图片在竖直方向上进行拼接。


# import os
# from PIL import Image, ImageDraw
# from PIL import ImageFont
# # 定义文件夹路径
# day_visible_folder = '/data/wangsong/datasets/CVC-14/Day/Visible/NewTest/FramesPos'
# day_infrared_folder = '/data/wangsong/datasets/CVC-14/Day/FIR/NewTest/FramesPos'
# night_visible_folder = '/data/wangsong/datasets/CVC-14/Night/Visible/NewTest/FramesPos'
# night_infrared_folder = '/data/wangsong/datasets/CVC-14/Night/FIR/NewTest/FramesPos'
# detection_result_folder = '/data/wangsong/results/CVC14/exp4/test/checkpoint/fusion_branch/tenth/det/'
# result_folder = './cvc14_result_in_paper/'

# # 确保结果文件夹存在
# os.makedirs(result_folder, exist_ok=True)

# # 绘制检测结果的函数
# def draw_detections(image_path, detections):
#     with Image.open(image_path) as img:
#         if img.mode != 'RGB' :
#             img = img.convert('RGB')
#         draw = ImageDraw.Draw(img)
#         font = ImageFont.load_default()
#         font_size = 12  # 字体大小
#         for det in detections:
#             # 解析检测结果的类别，坐标和置信度
#             category, x1, y1, x2, y2, score = det
#             if float(score) < 0.014 : continue 
#             # 绘制矩形框
#             draw.rectangle([float(x1), float(y1), float(x2), float(y2)], outline='green', width=2)
#             # # 可以在这里添加文本绘制代码，如果需要的话
#             # score_text = f"{float(score):.2f}"  # 保留两位小数
#             # # 计算文本位置
#             # text_x = float(x1)
#             # text_y = float(y1) - font_size  # 将文本放置在框的上方
#             # # 防止文本绘制在图片外面
#             # if text_y < 0:
#             #     text_y = float(y1)
#             # # 绘制文本
#             # draw.text((text_x, text_y), score_text, font=font, fill='red')
#         return img

# # 读取检测结果并绘制到图片上的函数
# def process_detection_file(detection_file):
#     with open(detection_file, 'r') as file:
#         detections = [line.strip().split() for line in file.readlines()]

#     # 获取文件名和时间段
#     filename = os.path.basename(detection_file)
#     time_of_day, image_name = filename.split('_', 1)
#     image_name = image_name.replace('.txt', '.tif')

#     # 根据时间段选择对应的文件夹
#     if time_of_day.lower() == 'day':
#         visible_img_path = os.path.join(day_visible_folder, image_name)
#         infrared_img_path = os.path.join(day_infrared_folder, image_name)
#     elif time_of_day.lower() == 'night':
#         visible_img_path = os.path.join(night_visible_folder, image_name)
#         infrared_img_path = os.path.join(night_infrared_folder, image_name)
#     else:
#         raise ValueError(f"Time of day '{time_of_day}' is not recognized.")

#     # 绘制检测结果
#     visible_img_annotated = draw_detections(visible_img_path, detections)
#     infrared_img_annotated = draw_detections(infrared_img_path, detections)

#     # 拼接图片
#     combined_img = Image.new('RGB', (visible_img_annotated.width, visible_img_annotated.height + infrared_img_annotated.height))
#     combined_img.paste(visible_img_annotated, (0, 0))
#     combined_img.paste(infrared_img_annotated, (0, visible_img_annotated.height))

#     # 保存结果图片
#     result_img_path = os.path.join(result_folder, image_name)
#     combined_img.save(result_img_path)

# # 遍历检测结果文件夹
# i = 0 
# for detection_file in os.listdir(detection_result_folder):
#     i += 1 
#     if i > 600 : break 
#     if detection_file.endswith('.txt'):
#         detection_file_path = os.path.join(detection_result_folder, detection_file)
#         process_detection_file(detection_file_path)

# print("Detection visualization completed.")




# import os
# from PIL import Image, ImageDraw

# # 定义文件夹路径
# day_visible_folder = '/data/wangsong/datasets/CVC-14/Day/Visible/NewTest/FramesPos'
# day_infrared_folder = '/data/wangsong/datasets/CVC-14/Day/FIR/NewTest/FramesPos'
# night_visible_folder = '/data/wangsong/datasets/CVC-14/Night/Visible/NewTest/FramesPos'
# night_infrared_folder = '/data/wangsong/datasets/CVC-14/Night/FIR/NewTest/FramesPos'
# detection_result_folder = '/data/wangsong/datasets/CVC-14/gt/CVC14_annotations/test-all/annotations'
# result_folder = './cvc14_gt_result/'

# # 确保结果文件夹存在
# os.makedirs(result_folder, exist_ok=True)

# # # 绘制检测结果的函数
# # def draw_detections(image_path, detections):
# #     with Image.open(image_path) as img:
# #         draw = ImageDraw.Draw(img)
# #         for det in detections:
# #             # 解析检测结果的类别，坐标和置信度
# #             category, x1, y1, w, h = det[:5]

# #             # 绘制矩形框
# #             draw.rectangle([float(x1), float(y1), float(w) + float(x1), float(h) + float(y1)], outline='red', width=2)
# #             # 可以在这里添加文本绘制代码，如果需要的话
# #         return img
# # 绘制检测结果的函数
# def draw_detections(image_path, detections):
#     with Image.open(image_path) as img:
#         if img.mode != 'RGB' :
#             img = img.convert('RGB')
#         draw = ImageDraw.Draw(img)
#         for det in detections:
#             # 解析检测结果的类别，坐标和置信度
#             category, x1, y1, w, h = det[:5]

#             # 绘制矩形框
#             draw.rectangle([float(x1), float(y1), float(w) + float(x1), float(h) + float(y1)], outline='yellow', width=2)
#         return img

# # 读取检测结果并绘制到图片上的函数
# def process_detection_file(detection_file):
#     with open(detection_file, 'r') as file:
#         detections = [line.strip().split() for line in file.readlines()]

#     del detections[0]
    
#     # 获取文件名和时间段
#     filename = os.path.basename(detection_file)
#     time_of_day, image_name = filename.split('_', 1)
#     image_name = image_name.replace('.txt', '.tif')

#     # 根据时间段选择对应的文件夹
#     if time_of_day.lower() == 'day':
#         visible_img_path = os.path.join(day_visible_folder, image_name)
#         infrared_img_path = os.path.join(day_infrared_folder, image_name)
#     elif time_of_day.lower() == 'night':
#         visible_img_path = os.path.join(night_visible_folder, image_name)
#         infrared_img_path = os.path.join(night_infrared_folder, image_name)
#     else:
#         raise ValueError(f"Time of day '{time_of_day}' is not recognized.")

#     # 绘制检测结果
#     visible_img_annotated = draw_detections(visible_img_path, detections)
#     infrared_img_annotated = draw_detections(infrared_img_path, detections)

#     # 拼接图片
#     combined_img = Image.new('RGB', (visible_img_annotated.width, visible_img_annotated.height + infrared_img_annotated.height))
#     combined_img.paste(visible_img_annotated, (0, 0))
#     combined_img.paste(infrared_img_annotated, (0, visible_img_annotated.height))

#     # 保存结果图片
#     result_img_path = os.path.join(result_folder, image_name)
#     combined_img.save(result_img_path)

# # 遍历检测结果文件夹
# i = 0 
# for detection_file in os.listdir(detection_result_folder):
#     i += 1 
#     if i > 600 : break 
#     if detection_file.endswith('.txt'):
#         detection_file_path = os.path.join(detection_result_folder, detection_file)
#         process_detection_file(detection_file_path)

# print("Detection visualization completed.")



import os
from PIL import Image, ImageDraw
from PIL import ImageFont
# 定义文件夹路径
day_visible_folder = '/data/wangsong/datasets/CVC-14/Day/Visible/NewTest/FramesPos'
day_infrared_folder = '/data/wangsong/datasets/CVC-14/Day/FIR/NewTest/FramesPos'
night_visible_folder = '/data/wangsong/datasets/CVC-14/Night/Visible/NewTest/FramesPos'
night_infrared_folder = '/data/wangsong/datasets/CVC-14/Night/FIR/NewTest/FramesPos'
# detection_result_folder = '/home/wangsong/exp/rgbt_ped_dect/Fusion-DETR-v1/evaluation_script/cvc_sota/mbnet_det'
detection_result_folder = '/data/wangsong/results/CVC14/exp4/test/checkpoint/fusion_branch/tenth/det/'
result_folder = '/home/wangsong/exp/rgbt_ped_dect/Fusion-DETR-v1/cvc14_result_msdetr_false_positive_blue'
gt_folder = '/data/wangsong/datasets/CVC-14/gt/CVC14_annotations/test-all/annotations'
# 确保结果文件夹存在
os.makedirs(result_folder, exist_ok=True)

# 绘制检测结果的函数
# def draw_detections(image_path, detections):
#     with Image.open(image_path) as img:
#         if img.mode != 'RGB' :
#             img = img.convert('RGB')
#         draw = ImageDraw.Draw(img)
#         font = ImageFont.load_default()
#         font_size = 12  # 字体大小
#         for det in detections:
#             # 解析检测结果的类别，坐标和置信度
#             category, x1, y1, x2, y2, score = det
#             if float(score) < 0.014 : continue 
#             # 绘制矩形框
#             draw.rectangle([float(x1), float(y1), float(x2), float(y2)], outline='green', width=2)
#             # # 可以在这里添加文本绘制代码，如果需要的话
#             # score_text = f"{float(score):.2f}"  # 保留两位小数
#             # # 计算文本位置
#             # text_x = float(x1)
#             # text_y = float(y1) - font_size  # 将文本放置在框的上方
#             # # 防止文本绘制在图片外面
#             # if text_y < 0:
#             #     text_y = float(y1)
#             # # 绘制文本
#             # draw.text((text_x, text_y), score_text, font=font, fill='red')
#         return img
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # 计算交集部分
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    # 计算交集面积
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)

    # 计算并集面积
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = box1_area + box2_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area

    return iou

def draw_detections(image_path, detections, gt_boxes, iou_threshold=0.5):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        font_size = 12  # 字体大小
        for det in detections:
            category, x1, y1, x2, y2, score = det
            x1, y1, x2, y2 = float(x1), float(y1), float(x2) , float(y2)
           
            if float(score) < 0.014:
                continue
            color = 'green'
            for gt_box in gt_boxes:
                gt_x, gt_y, gt_w, gt_h = gt_box
                gt_x1, gt_y1, gt_x2, gt_y2 = float(gt_x), float(gt_y), float(gt_x) + float(gt_w), float(gt_y) + float(gt_h)
                iou = calculate_iou((x1, y1, x2, y2), (gt_x1, gt_y1, gt_x2, gt_y2))
                if iou >= iou_threshold:  
                    color = 'green'
                    break
                else:
                    color = 'blue'
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        return img


# 读取检测结果并绘制到图片上的函数
def process_detection_file(detection_file , gt_file):
    with open(detection_file, 'r') as file:
        detections = [line.strip().split() for line in file.readlines()]

    # 获取文件名和时间段
    filename = os.path.basename(detection_file)
    time_of_day, image_name = filename.split('_', 1)
    image_name = image_name.replace('.txt', '.tif')

    # 根据时间段选择对应的文件夹
    if time_of_day.lower() == 'day':
        visible_img_path = os.path.join(day_visible_folder, image_name)
        infrared_img_path = os.path.join(day_infrared_folder, image_name)
    elif time_of_day.lower() == 'night':
        visible_img_path = os.path.join(night_visible_folder, image_name)
        infrared_img_path = os.path.join(night_infrared_folder, image_name)
    else:
        raise ValueError(f"Time of day '{time_of_day}' is not recognized.")

    # 读取真值标注文件
    with open(gt_file, 'r') as file:
        gt_boxes = [line.strip().split() for line in file.readlines()]
        del gt_boxes[0]
        gt_boxes = [(float(box[1]), float(box[2]), float(box[3]), float(box[4])) for box in gt_boxes]
        
    # 绘制检测结果
    visible_img_annotated = draw_detections(visible_img_path, detections , gt_boxes)
    infrared_img_annotated = draw_detections(infrared_img_path, detections , gt_boxes)

    # 拼接图片
    combined_img = Image.new('RGB', (visible_img_annotated.width, visible_img_annotated.height + infrared_img_annotated.height))
    combined_img.paste(visible_img_annotated, (0, 0))
    combined_img.paste(infrared_img_annotated, (0, visible_img_annotated.height))

    # 保存结果图片
    result_img_path = os.path.join(result_folder, image_name)
    combined_img.save(result_img_path)

# 遍历检测结果文件夹
i = 0 
for detection_file in os.listdir(detection_result_folder):
    i += 1 
    if i > 600 : break 
    if detection_file.endswith('.txt'):
        detection_file_path = os.path.join(detection_result_folder, detection_file)
        gt_file = os.path.join(gt_folder,detection_file)
        process_detection_file(detection_file_path,gt_file)

print("Detection visualization completed.")

