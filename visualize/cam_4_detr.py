from visualize.pytorch_grad_cam import AblationCAM
from visualize.pytorch_grad_cam.ablation_layer import AblationLayer, AblationLayer4DETREncoderRGB, AblationLayer4DETREncoderT
from visualize.pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from visualize.pytorch_grad_cam.utils.reshape_transforms import detr_backbone_reshape_transform, detr_encoder_rgb_reshape_transform, detr_encoder_t_reshape_transform
from visualize.pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
from pathlib import Path
import numpy as np
import cv2, os
from util.misc import nested_tensor_from_tensor_list

coco_names = ['person', ]
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
cam_list = list(range(10))


def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image


def save_result_image(image_float_np, grayscale_cam, boxes, labels, classes, path):
    image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
    image = draw_boxes(boxes, labels, classes, image)
    image = Image.fromarray(image)

    image.save(path)


def detr_cam(model, dataset, output_dir):
    output_dir = os.path.join(output_dir, 'cam')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ind in cam_list:
        input_rgb, input_t, result = dataset[ind]
        input_rgb = input_rgb.unsqueeze(0)
        input_t = input_t.unsqueeze(0)
        input_rgb = nested_tensor_from_tensor_list(input_rgb).to('cuda')
        input_t = nested_tensor_from_tensor_list(input_t).to('cuda')
        outputs = model(input_rgb, input_t)
        labels = outputs[0]['labels'].to('cpu')
        boxes = outputs[0]['boxes'].to('cpu')
        if boxes.shape[0] == 0:
            continue
        classes = ['person' for i in range(len(labels))]

        img_path_rgb = result['img_absolute_path_rgb']
        img_path_t = result['img_absolute_path_t']

        image_rgb = np.array(img_path_rgb)
        image_t = np.array(img_path_t)
        image_float_np_rgb = np.float32(image_rgb) / 255
        image_float_np_t = np.float32(image_t) / 255

        grayscale_cam_1 = detr_cam_4_backbone_rgb(model, labels, boxes, input_rgb, input_t)
        path_1 = os.path.join(output_dir, str(ind) + '_1.jpg')

        grayscale_cam_2 = detr_cam_4_backbone_t(model, labels, boxes, input_rgb, input_t)
        path_2 = os.path.join(output_dir, str(ind) + '_2.jpg')

        grayscale_cam_3 = detr_cam_4_encoder_rgb(model, labels, boxes, input_rgb, input_t)
        path_3 = os.path.join(output_dir, str(ind) + '_3.jpg')

        grayscale_cam_4 = detr_cam_4_encoder_t(model, labels, boxes, input_rgb, input_t)
        path_4 = os.path.join(output_dir, str(ind) + '_4.jpg')

        save_result_image(image_float_np_rgb, grayscale_cam_1, boxes, labels, classes, path_1)
        save_result_image(image_float_np_t, grayscale_cam_2, boxes, labels, classes, path_2)
        save_result_image(image_float_np_rgb, grayscale_cam_3, boxes, labels, classes, path_3)
        save_result_image(image_float_np_t, grayscale_cam_4, boxes, labels, classes, path_4)


def detr_cam_4_backbone_rgb(model, labels, boxes, input_rgb, input_t):
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    target_layers = [model.backbone_rgb]
    cam = AblationCAM(model, target_layers, use_cuda=True,
                      reshape_transform=detr_backbone_reshape_transform,
                      ablation_layer=AblationLayer(),
                      ratio_channels_to_ablate=1.0)

    grayscale_cam = cam(input_rgb, input_t, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    return grayscale_cam


def detr_cam_4_backbone_t(model, labels, boxes, input_rgb, input_t):
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    target_layers = [model.backbone_t]
    cam = AblationCAM(model, target_layers, use_cuda=True,
                      reshape_transform=detr_backbone_reshape_transform,
                      ablation_layer=AblationLayer(),
                      ratio_channels_to_ablate=1.0)

    grayscale_cam = cam(input_rgb, input_t, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    return grayscale_cam


def detr_cam_4_encoder_rgb(model, labels, boxes, input_rgb, input_t):
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    target_layers = [model.fusion_transformer.encoder]
    cam = AblationCAM(model, target_layers, use_cuda=True,
                      reshape_transform=detr_encoder_rgb_reshape_transform,
                      ablation_layer=AblationLayer4DETREncoderRGB(),
                      ratio_channels_to_ablate=1.0)

    grayscale_cam = cam(input_rgb, input_t, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    return grayscale_cam


def detr_cam_4_encoder_t(model, labels, boxes, input_rgb, input_t):
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    target_layers = [model.fusion_transformer.encoder]
    cam = AblationCAM(model, target_layers, use_cuda=True,
                      reshape_transform=detr_encoder_t_reshape_transform,
                      ablation_layer=AblationLayer4DETREncoderT(),
                      ratio_channels_to_ablate=1.0)

    grayscale_cam = cam(input_rgb, input_t, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    return grayscale_cam