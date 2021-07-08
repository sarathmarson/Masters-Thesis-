import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import math
import glob
import tensorflow as tf
from collections import defaultdict
import sys
from scipy.spatial import distance
import matplotlib.pyplot as plt
import itertools
from yolov3_tf2.models import (YoloV3, YoloV3Tiny)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from utils import draw_outputs
from chessboard_detection_debug_cells import detectChessBoard as cb

#flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('classes', './data/labels/obj.names', 'path to classes file')
#flags.DEFINE_string('weights', './weights/yolov3.tf','path to weights file')
flags.DEFINE_string('weights', './weights/yolo_solo_set.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_list('images', '/data/images/dog.jpg', 'list with paths to input images')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
#flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('num_classes', 12, 'number of classes in the model')

calibrate = True


def main(_argv):
  
  calibrate = True

  print('weights loaded')

  #image_ = cv2.imread('/home/sarath/Object-Detection-API/dataset/calibrated/chessboard_hd_1/dataset/image/test/imageFiles/frame1624897799cal.png')
  image_ = cv2.imread('/home/sarath/Object-Detection-API/dataset/calibrated/h/frame1624966914cal.png')
  # image_ = cv2.cvtColor(image_, cv2.COLOR_RGB2BGR)

  if calibrate is True:

    ret, mtx, dist, rvecs, tvecs = cb.calibration()
    h,  w = image_.shape[:2]
    print(h,w)
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    calibrate = False


  chess_cell_points,_,_ = cb.find_chess_board_cells(image_, mtx, dist)
  
  x_points_obtained_ , y_points_obtained_ = cb.extract_xy(chess_cell_points)  

  x_points_obtained_ = cb.group_split(x_points_obtained_, 9)
  y_points_obtained_ = cb.group_split(y_points_obtained_, 9)

  # print(x_points_obtained_)
  # print(y_points_obtained_)

  cell_full = cb.get_chess_cells_array(x_points_obtained_, y_points_obtained_)

  # print(cell_full)

  print('cells detected')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
