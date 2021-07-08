from absl import logging
import numpy as np
import tensorflow as tf
import cv2
from seaborn import color_palette
from PIL import Image, ImageDraw, ImageFont
from chess_cells_detection import detectChessBoard as ccd

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]




def draw_outputs(img, outputs, class_names,cell_full):
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='./data/fonts/futur.ttf',
                              size=(img.size[0] + img.size[1]) // 100)
    for i in range(nums):
        color = colors[int(classes[i])]
        x1y1 = ((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = ((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        width_box, height_box = wh
        test_boxes = np.array(boxes[i])
        test_point = [[[int(test_boxes[0]*width_box) , int(test_boxes[1]*height_box)]], [[int(test_boxes[2]*width_box), int(test_boxes[3]*height_box)]]]

        #print(width_box, height_box)

        cell_name = ccd.get_cell_name(test_point, cell_full,wh)

        thickness = (img.size[0] + img.size[1]) // 200
        x0, y0 = x1y1[0], x1y1[1]
        for t in np.linspace(0, 1, thickness):
            x1y1[0], x1y1[1] = x1y1[0] - t, x1y1[1] - t
            x2y2[0], x2y2[1] = x2y2[0] - t, x2y2[1] - t
            draw.rectangle([x1y1[0], x1y1[1], x2y2[0], x2y2[1]], outline=tuple(color))
        confidence = '{:.2f}%'.format(objectness[i]*100)
        text = '{} {} {}'.format(class_names[int(classes[i])], confidence, cell_name)
        text_size = draw.textsize(text, font=font)
        draw.rectangle([x0, y0 - text_size[1], x0 + text_size[0], y0],
                        fill=tuple(color))
        draw.text((x0, y0 - text_size[1]), text, fill='black',
                              font=font)
    rgb_img = img.convert('RGB')
    img_np = np.asarray(rgb_img)
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    return img

