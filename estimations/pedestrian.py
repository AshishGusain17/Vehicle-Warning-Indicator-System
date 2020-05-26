import numpy as np
import os
import sys
import tensorflow as tf
from imutils.video import VideoStream
import cv2
import imutils
import time
from imutils.video import FPS
from sklearn.metrics import pairwise
import copy
import pathlib
from collections import defaultdict



from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util



utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
PATH_TO_LABELS = '../bigdata/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)




model_name = 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
model_dir =  "../bigdata/models/" + model_name + "/saved_model"
detection_model = tf.saved_model.load(str(model_dir))
detection_model = detection_model.signatures['serving_default']



# print(category_index)
print(detection_model.inputs)
print(detection_model.output_dtypes)
print(detection_model.output_shapes)

font = cv2.FONT_HERSHEY_PLAIN

flag = 0
area = 0
def estimate_stepping(output_dict,height,width,image_np):
  pedes_present = 0
  global flag,area

  for ind,scr in enumerate(output_dict['detection_classes']):
    if scr==1:
      ymin, xmin, ymax, xmax = output_dict['detection_boxes'][ind]
      score = output_dict['detection_scores'][ind]
      if score>0.4:
        # cv2.putText(image_np,'min xy',(int(xmin*width),int(ymin*height)), font, 1,(255,0,0),2,cv2.LINE_AA)
        # cv2.putText(image_np,'max xy',(int(xmax*width),int(ymax*height)), font, 1,(255,0,0),2,cv2.LINE_AA)
        area = int((xmax - xmin)*width * (ymax - ymin)*height)
        print(output_dict['detection_boxes'][ind],output_dict['detection_scores'][ind],area)
        if area>2500:
          pedes_present = 1
          flag=12
  if pedes_present == 0:
    flag=flag-1


  if flag > 0:
    if area > 15000:
      cv2.putText(image_np,"STOP IT !!! DON'T HIT HIM " + str(area),(50,50), font, 3,(255,255,0),2,cv2.LINE_AA)
    else:
      cv2.putText(image_np,"BE CAREFUL !!! Someone is in front " + str(area),(50,50), font, 3,(255,255,0),2,cv2.LINE_AA)







def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]

  # output_dict is a dict  with keys detection_classes , num_detections , detection_boxes(4 coordinates of each box) , detection_scores for 100 boxes
  output_dict = model(input_tensor)


  # num_detections gives number of objects in current frame
  num_detections = int(output_dict.pop('num_detections'))


  # output_dict is a dict  with keys detection_classes , detection_boxes(4 coordinates of each box) , detection_scores for num_detections boxes
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  # adding num_detections that was earlier popped out
  output_dict['num_detections'] = num_detections
  # converting all values in detection_classes as ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
  # print(5,output_dict)


  # Handle models with masks:
  # if 'detection_masks' in output_dict:
  #   # Reframe the the bbox mask to the image size.
  #   detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
  #             output_dict['detection_masks'], output_dict['detection_boxes'],
  #              image.shape[0], image.shape[1])   
  #   detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
  #                                      tf.uint8)
  #   output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
  return output_dict





def show_inference(model, image_path):
  image_np = np.array(image_path)
  height,width,channel = image_np.shape
  print(image_np.shape)

  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)

  estimate_stepping(output_dict,height,width,image_np)
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









# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('../videos/a.mp4')
time.sleep(2.0)
cap.set(1,913*25)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out1 = cv2.VideoWriter('i.avi', fourcc, 3.0, (int(cap.get(3)),int(cap.get(4))))
fps = FPS().start()
ctt = 0
while True:
    (grabbed, frame) = cap.read()
    print('frame',frame.shape)

    # print(ctt)
    # ctt = ctt + 1
    # if ctt==334:
    #   break
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








# a.mp4   104*25   843*25      913*25