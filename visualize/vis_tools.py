import os
import PIL
import operator

import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
import matplotlib.patches
import cv2

from time import sleep
from tqdm import tqdm
from eval.bbGt import get_annotation_version, get_field_number


def drawBBoxes(img, bboxes, bboxes_mode='xywh', percentile=False, colors=None, widths=None, descriptions=None, saved_path=None):
    """
    在图像上标记bounding boxes, 注意生成图像中会包含坐标轴以及额外的空白区域，故而生成图像不能再次画框

    :param img: 单张图像或图像的绝对路径
    :param bboxes: List，存放bounding box数据，结构为[[x_upper_left, y_upper_left, w, h],...]
    :param bboxes_mode: 边界框的格式，支持'xywh'和'xyxy'两种
    :param percentile: 边界框的坐标值是否为百分比形式的数据
    :param colors: str或者list
    :param widths: float或者list
    :param descriptions: str或者list
    :param saved_path: 处理之后图像存放的路径，如果None，则会在线展示处理后的图像
    """
    assert bboxes_mode in ('xywh', 'xyxy')
    if isinstance(img, str):
        assert os.path.isfile(img)
        img = PIL.Image.open(img)
        img_W, img_H = img.size
    else:
        img_H, img_W = img.shape[:2]

    pyplot.imshow(img)
    pyplot.axis('off')
    img = pyplot.gca()

    if bboxes is not None:
        boxes_length = len(bboxes)

        if colors is None:
            colors = ['red'] * boxes_length
        elif isinstance(colors, str):
            colors = [colors] * boxes_length
        elif isinstance(colors, list):
            assert len(colors) == len(bboxes)
        else:
            raise RuntimeError

        if widths is None:
            widths = [1.5] * boxes_length
        elif isinstance(widths, float):
            widths = [widths] * boxes_length
        elif isinstance(widths, list):
            assert len(widths) == boxes_length
        else:
            raise RuntimeError

        if descriptions is None:
            descriptions = [''] * boxes_length
        elif isinstance(descriptions, str):
            descriptions = [descriptions] * boxes_length
        elif isinstance(descriptions, list):
            assert len(descriptions) == boxes_length
        else:
            raise RuntimeError

        for bbox, color, width, description in zip(bboxes, colors, widths, descriptions):
            if bboxes_mode == 'xyxy':
                x_min, y_min, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            else:
                x_min, y_min, w, h = bbox[0] - 0.5 * bbox[2], bbox[1] - 0.5 * bbox[3], bbox[2], bbox[3]
            if percentile:
                rect = matplotlib.patches.Rectangle((x_min * img_W, y_min * img_H), w * img_W, h * img_H,
                                                    edgecolor=color, linewidth=width, facecolor='none')
            else:
                rect = matplotlib.patches.Rectangle((x_min, y_min), w, h, edgecolor=color, linewidth=width,
                                                    facecolor='none')
            img.add_patch(rect)

            if description is not None:
                img.text(x_min, y_min, s=description, color=color, verticalalignment='bottom',  bbox={'alpha': 1})

    if saved_path is not None:
        saved_dir = os.path.dirname(saved_path)
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)
        pyplot.savefig(saved_path, bbox_inches='tight', pad_inches=0)
    else:
        pyplot.show()

    pyplot.close()


if __name__ == '__main__':
    t_img_path = '/data/wangsong/datasets/KAIST/Images/set03/V000/lwir/I01418.jpg'
    rgb_img_path = '/data/wangsong/datasets/KAIST/Images/set03/V000/visible/I01418.jpg'

    t_boxes = [[51, 200, 110, 359], [585, 225, 630, 346], [159, 208, 213, 331]]
    drawBBoxes(t_img_path, t_boxes, bboxes_mode='xyxy', widths=1.5, colors='red', descriptions='', saved_path='/home/wangsong/temp.jpg')
