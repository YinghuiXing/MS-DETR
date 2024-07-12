

'''
            # 可见光路径 targets['img_absolute_path_rgb']
            # 红外 target['img_absolute_path_t']
            print(targets)
            print(outputs.keys())
            print("pred_logits" , pred_logits.shape)
            print("pred_boxes" , pred_boxes.shape) 
            print("points_list" , len(points_list) , points_list[0].shape) 
            print("weights_list" ,  len(weights_list) , weights_list[0].shape) 
            
img_rgb_path ['/data/wangsong/datasets/KAIST/Images/set10/V000/visible/I01219.jpg', '/data/wangsong/datasets/KAIST/Images/set06/V001/visible/I00019.jpg']
img_t_path ['/data/wangsong/datasets/KAIST/Images/set10/V000/lwir/I01219.jpg', '/data/wangsong/datasets/KAIST/Images/set06/V001/lwir/I00019.jpg']
pred_logits torch.Size([2, 300, 2])
pred_boxes torch.Size([2, 300, 4])
points_list 6 torch.Size([2, 300, 8, 8, 4, 2])
weights_list 6 torch.Size([2, 300, 8, 8, 4])

'''
import os
import torch
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tifffile as tf
# show_points(input_point, input_label, plt.gca())
def show_points(coords , ax, marker_size=375):
    pos_points = coords[labels==1]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    # ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    

def apply_min_max_normalization(points):
    # 每个点是 (x,y)，我们对x和y分别归一化
    x_points = points[..., 0]
    y_points = points[..., 1]

    # 找到x和y的最小值和最大值
    x_min, x_max = x_points.min(), x_points.max()
    y_min, y_max = y_points.min(), y_points.max()

    # 应用MIN-MAX归一化
    x_normalized = (x_points - x_min) / (x_max - x_min)
    y_normalized = (y_points - y_min) / (y_max - y_min)

    # 将归一化后的x和y重新组合成点坐标
    normalized_points = torch.stack((x_normalized, y_normalized), dim=-1)
    return normalized_points

def apply_min_max_normalization_weight(weight):
    # 每个点是 (x,y)，我们对x和y分别归一化
    min_weight = weight.min()
    max_weight = weight.max()

    normalized_weights_flat = (weight - min_weight) / (max_weight - min_weight)

    return normalized_weights_flat


# def draw_point(pred_logits  , points , weights , img_rgb_path_list , img_t_path_list , save_dir='/data/wangsong/results/24/draw_point/kaist'):
#     # print("img_rgb_path" , img_rgb_path_list)
#     # print("img_t_path" ,img_t_path_list)
#     # print("pred_logits" , pred_logits.shape)
#     # print("pred_boxes" , pred_boxes.shape) 
#     # print("points_list" , len(points) , points[0].shape) 
#     # print("weights_list" ,  len(weights) , weights[0].shape) 
#     bs = len(img_rgb_path_list)
#     for i in range(bs) :
#         # 取出当前样本的预测置信度  选点  点权重
#         cur_rgb = img_rgb_path_list[i] 
#         cur_t = img_t_path_list[i]
#         cur_pred_logits = pred_logits[i] # torch.Size([300, 2]) 我不理解
#         cur_points = points[i] #  torch.Size([300, 8, 8, 4, 2])
#         cur_weight = weights[i] # torch.Size([300, 8, 8, 4])
#         # print(cur_pred_logits.shape)
#         # print(torch.topk(cur_pred_logits[:,0],10))
#         # print(torch.topk(cur_pred_logits[:,1],10))
#         # print(cur_points.shape)
#         # print(cur_weight.shape)
        
#         '''
#             思路是选取置信度最高的query
#             此时这个query对应8*8*4*2个点  共计256个点
#             8个头
#             8个特征层级
#             4 每个特征层级采样四个点
            
#         '''
#         # 使用置信度最高的query点 (前10)
#         topk_vals, topk_inds = torch.topk(cur_pred_logits[:, 0], 2)
        
#         # 打开图像文件
#         img_rgb = Image.open(cur_rgb).convert('RGB')
#         img_t = Image.open(cur_t).convert('RGB')  # 红外图像可能是单通道的，这里确保它是RGB
        
#         # 获取图像的大小，以便将归一化的点坐标转换为像素坐标
#         width, height = img_rgb.size
        
#         # 绘制点在 RGB 和 红外图像上
#         for idx in topk_inds:
#             # 提取cur_points和cur_weight中对应于idx的点
#             point_set = cur_points[idx].reshape(-1, 2)  # 变更形状到 (8*8*4, 2)
#             weight_set = cur_weight[idx].reshape(-1)
#             point_set = apply_min_max_normalization(point_set)
#             for point, weight in zip(point_set, weight_set):
#                 x = int(point[0].item() * width)
#                 y = int(point[1].item() * height)
#                 # print(x,y)
#                 # 使用点的权重作为圆的半径
#                 radius = int(weight.item() * 10)  # 可以根据需要调整这个放大因子
#                 draw_rgb = ImageDraw.Draw(img_rgb)
#                 draw_t = ImageDraw.Draw(img_t)
#                 # 绘制一个圆来表示点
#                 draw_rgb.ellipse((x - radius, y - radius, x + radius, y + radius), outline='red', width=3)
#                 draw_t.ellipse((x - radius, y - radius, x + radius, y + radius), outline='red', width=3)
        
#         # 确保保存目录存在
#         if not os.path.exists(os.path.join(save_dir, "rgb")):
#             os.makedirs(os.path.join(save_dir, "rgb"))
#         if not os.path.exists(os.path.join(save_dir, "t")):
#             os.makedirs(os.path.join(save_dir, "t"))
        
#         # 保存绘制了点的图像
#         rgb_path, _ = os.path.splitext(cur_rgb)
#         rgb_path = "_".join(rgb_path.split('/')[3:])
        
#         t_path, _ = os.path.splitext(cur_t)
#         t_path = "_".join(t_path.split('/')[3:])

#         # print(os.path.join(save_dir, f'{cur_rgb}.png'))
#         img_rgb_save_path = os.path.join(save_dir, "rgb" ,  f'{rgb_path}.jpg')
#         img_t_save_path = os.path.join(save_dir, "t" ,  f'{t_path}.jpg')
#         img_rgb.save(img_rgb_save_path)
#         img_t.save(img_t_save_path)

# pred_logits torch.Size([2, 300, 2]) 
# torch.Size([2, 300, 8, 8, 4, 2])  ->  2 300 8 4 4 2  rgb |  2 300 8 4 4 2   t
def draw_point(pred_logits, points, weights, img_rgb_path_list, img_t_path_list, save_dir='/data/wangsong/results/24/draw_point/kaist_redstar'):
    bs = len(img_rgb_path_list)
     # 确保保存目录存在
    if not os.path.exists(os.path.join(save_dir, "rgb")):
        os.makedirs(os.path.join(save_dir, "rgb"))
    if not os.path.exists(os.path.join(save_dir, "t")):
        os.makedirs(os.path.join(save_dir, "t"))
        
        
    for i in range(bs):  # 枚举图片  每一次迭代都是一对图片
        # 对于当前图片，获取300个query的预测值  其中0维是前景
        cur_pred_logits = pred_logits[i].sigmoid()  #  300, 2
        
        cur_points =  points[i]   # points 形状 torch.Size([2, 300, 8, 8, 4, 2])  获取当前样本的points 300,8,8,4,2
        cur_weight = weights[i]   # 同上，每一个权重对应一个点  300,8,8,4
        # _ , topk_inds = torch.topk(cur_pred_logits[:, 0], 10)
        mask = cur_pred_logits[:, 0] > 0.2  # 创建一个布尔掩码，表示哪些元素满足条件

        topk_inds = torch.nonzero(mask).squeeze(1)  # 获取满足条件的元素的索引

        # 对于RGB和红外图，进行操作
        # 首先分别获取两个模态对应的点和权重
        # rgb_points, t_points = torch.chunk(cur_points, chunks=2, dim=3)
        cur_points_list = torch.chunk(cur_points, chunks=2, dim=2)
        # rgb_weight, t_weight = torch.chunk(cur_weight, chunks=2, dim=3)
        cur_weight_list = torch.chunk(cur_weight, chunks=2, dim=2)
        img_paths = {'rgb': img_rgb_path_list[i], 't': img_t_path_list[i]}
        for j , img_type in enumerate(['rgb', 't']):
            img_path = img_paths[img_type]
            specified_modality_points  = cur_points_list[j]  # 获取对应模态的点    300, 8, 4, 4, 2
            specified_modality_weights = cur_weight_list[j]  # 获取对应模态的点的权重  300, 8, 4, 4
            # 读取图像  这是应对cvc14读取通道不对的情况  kaist不需要 直接Image.open就行
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tmp = Image.open(img_path)
            width, height = img_tmp.size

            pos_points = []
            marker_size = []
            
            # 我们使用 subplots 方法来创建一个新的图像
            fig, ax = plt.subplots(tight_layout=True)

            # 显示图像
            if img_type == 'rgb' :
                ax.imshow(img)
            else :
                ax.imshow(img, cmap='gray')

            # 获取点坐标和对应权重
            for k in topk_inds:
                p = specified_modality_points[k].reshape(-1, 2)  # ([8*4*4, 2])
                w = specified_modality_weights[k].reshape(-1)  # 8*4*4

                for point, weight in zip(p, w):
                    if weight.item() > 0.5 : 
                        pos_points.append([point[0].item() * width, point[1].item() * height])
                        marker_size.append((weight.item() * 10)**2)
            pos_points = torch.tensor(pos_points)
            if len(pos_points) > 0:
                ax.scatter(pos_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='red', linewidth=0.8)


            # 去掉坐标轴
            plt.axis('off')

            # 配置保存路径
            save_path_prefix = os.path.join(save_dir, img_type , '_'.join(img_path.split('/')[-4:]))
            save_path = os.path.splitext(save_path_prefix)[0]  + f'_{img_type}.jpg'

            # 保存绘制了点的图像
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

            # 关闭图形，释放内存
            plt.close()