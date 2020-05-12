import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import pathlib
from collections import defaultdict
from matplotlib import pyplot as plt
import cv2
import imutils
import time
from sklearn.metrics import pairwise
from imutils.video import FPS



from utils import ops as utils_ops
from utils import label_map_util



utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
PATH_TO_LABELS = '../bigdata/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)




model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09'
model_dir =  "../bigdata/models/" + model_name + "/saved_model"
detection_model = tf.saved_model.load(str(model_dir))
detection_model = detection_model.signatures['serving_default']



# print(category_index)
colors = np.random.uniform(0, 255, size=(len(category_index), 3))
font = cv2.FONT_HERSHEY_PLAIN

print(detection_model.inputs)
print(detection_model.output_dtypes)
print(detection_model.output_shapes)









def visualize(output_dict,image_np,height,width):
  class_ids = []
  confidences = []
  boxes = []
  num = output_dict['num_detections']
  for ind in range(num):
    scr = output_dict['detection_scores'][ind]
    classId = output_dict['detection_classes'][ind] 
    box = output_dict['detection_boxes'][ind]

    ymin, xmin, ymax, xmax = box
    confidences.append(float(scr))
    class_ids.append(classId)
    boxes.append([int(xmin*width) , int(ymin*height) , int((xmax-xmin)*width) , int((ymax-ymin)*height)])

  indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
  # if len(boxes) != len(indexes):
  #   print(indexes,boxes , confidences,class_ids)
  for j in indexes:
    i = j[0]
    x, y, w, h = boxes[i]
    label = category_index[class_ids[i]]['name']
    color = colors[i]
    cv2.rectangle(image_np, (x, y), (x + w, y + h), color, 3)
    cv2.putText(image_np, label, (x, y - 5), font, 3, color, 3)
  # return image_np



def show_inference(model, image_path):
  image_np = np.array(image_path)
  # print(image_np.shape)
  height,width,channel = image_np.shape
  input_tensor = tf.convert_to_tensor(image_np)
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


  visualize(output_dict,image_np,height,width)
  cv2.imshow("version", image_np)




# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('../videos/a.mp4')
cap.set(1,190*25)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out1 = cv2.VideoWriter('i.avi', fourcc, 3.0, (int(cap.get(3)),int(cap.get(4))))

fps = FPS().start()

ctt = 0
while True:
    (grabbed, frame) = cap.read()
    print(ctt)
    ctt = ctt + 1

    show_inference(detection_model, frame)


    # cv2.imshow("version2", frame)
    # out1.write(frame)
    fps.update()
    key=cv2.waitKey(100)
    if key & 0xFF == ord("q"):
        break
        
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
# out1.release()
cv2.destroyAllWindows() 
