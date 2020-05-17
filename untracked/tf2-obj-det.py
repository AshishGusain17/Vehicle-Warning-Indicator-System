import numpy as np
import os
import sys
import tensorflow as tf
from imutils.video import VideoStream
import six.moves.urllib as urllib
import cv2
import imutils
import time
from imutils.video import FPS
from sklearn.metrics import pairwise
import copy
import pathlib
from collections import defaultdict


sys.path.append('../')


from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util



utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
PATH_TO_LABELS = '../../bigdata/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)




model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09'
model_dir =  "../../bigdata/models/" + model_name + "/saved_model"
detection_model = tf.saved_model.load(str(model_dir))
detection_model = detection_model.signatures['serving_default']



# print(category_index)
print(detection_model.inputs)
print(detection_model.output_dtypes)
print(detection_model.output_shapes)




def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]



  # output_dict is a dict  with keys detection_classes , num_detections , detection_boxes(4 coordinates of each box) , detection_scores for 100 boxes
  output_dict = model(input_tensor)
  # print(1,output_dict)




  # num_detections gives number of objects in current frame
  num_detections = int(output_dict.pop('num_detections'))
  # print(2,num_detections)



  # output_dict is a dict  with keys detection_classes , detection_boxes(4 coordinates of each box) , detection_scores for num_detections boxes
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}


  # adding num_detections that was earlier popped out
  output_dict['num_detections'] = num_detections


  # converting all values in detection_classes as ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
  print(5,output_dict)



  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])   
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    print(5,detection_masks_reframed) 
  return output_dict





def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  # image_np = np.array(Image.open(image_path))
  image_np = np.array(image_path)
  print(image_np.shape)

  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  return image_np








from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from sklearn.metrics import pairwise
from imutils.video import FPS

# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('../../videos/a.mp4')
time.sleep(2.0)

cap.set(1,190*25)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out1 = cv2.VideoWriter('i.avi', fourcc, 3.0, (int(cap.get(3)),int(cap.get(4))))

fps = FPS().start()

ctt = 0
while True:
    (grabbed, frame) = cap.read()
    print('frame',frame.shape)

    # resize the frame, blur it, and convert it to the HSV color space
    # frame = imutils.resize(frame, width=600)
    # print('frame_resize',frame.shape)
    print(ctt)
    ctt = ctt + 1
    if ctt==334:
      break
    frame=show_inference(detection_model, frame)


    cv2.imshow("version", frame)
    # out1.write(frame)
    fps.update()

    key=cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break
        
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
# out1.release()
cv2.destroyAllWindows() 


