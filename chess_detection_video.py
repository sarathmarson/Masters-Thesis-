from itertools import filterfalse
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
#from yolov3_tf2.utils import draw_outputs
from utils import draw_outputs
from chess_cells_detection import detectChessBoard as ccd


#flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('classes', './data/labels/obj.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_string('video', './data/video/paris.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 12, 'number of classes in the model')


def main(_argv):

    calibrate = True
    debug_mode = False

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        vid.set(cv2.CAP_PROP_FPS, 2)
    except:
        vid = cv2.VideoCapture(FLAGS.video)
    


    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    fps = 0.0
    count = 0
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        '''
        ********calibration and undistorting********

        calibrating camera using calibration images.
        This has to be performed only once.
        '''

        if calibrate:
            ret, mtx, dist, rvecs, tvecs = ccd.calibration()
            h,  w = img_in.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
            calibrate = False

        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        img_undistort = cv2.undistort(img_in, mtx, dist, None, newcameramtx)
        x1,y1,w1,h1 = roi
        img_cropped = img_undistort[y1:y1+h1, x1:x1+w1]
        img_in_ccd = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR)

        '''
        Now, img_cropped is the input for detection with yolo network and chess cells detection.
        '''

        chess_cell_points= ccd.find_chess_board_cells(img_in_ccd, debug_mode)
  
        x_points_obtained_ , y_points_obtained_ = ccd.extract_xy(chess_cell_points)  

        x_points_obtained_ = ccd.group_split(x_points_obtained_, 9)
        y_points_obtained_ = ccd.group_split(y_points_obtained_, 9)

        #print('x:  ',x_points_obtained_)
        #print('y:  ', y_points_obtained_)

        cell_full = ccd.get_chess_cells_array(x_points_obtained_, y_points_obtained_)      
        
        img_in1 = tf.expand_dims(img_cropped, 0)
        img_in2 = transform_images(img_in1, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in2)
        fps  = ( fps + (1./(time.time()-t1)) ) / 2

        img_to_display = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR) 
        #img1 = draw_outputs(img_to_display, (boxes, scores, classes, nums), class_names)
        img1 = draw_outputs(img_in_ccd, (boxes, scores, classes, nums), class_names,cell_full)
        img1 = cv2.putText(img1, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        
        if FLAGS.output:
            out.write(img1)
        cv2.imshow('output', img1)
        if cv2.waitKey(1) == ord('q'):
            break
        
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
