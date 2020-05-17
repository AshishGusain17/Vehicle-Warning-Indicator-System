import numpy as np
import cv2
import imutils
import time
from sklearn.metrics import pairwise
from imutils.video import FPS
import copy
import os
import sys
import tensorflow as tf
import pathlib
from collections import defaultdict



from utils import tracking_utils 
from utils import signalDetection_utils 
from utils import estimate_collide_utils
from utils import estimate_stepping_utils
from utils import lane_detection_utils
from utils import functions

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

  




def click_and_crop(event, x, y, flags, param):
  global refPt
  # if the left mouse button was clicked, record the starting (x, y) coordinates
  if event == cv2.EVENT_LBUTTONDOWN:
    refPt.append([x, y])



def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, [255,255,255])
    # cv2.imshow("mask",mask)
    masked = cv2.bitwise_and(img, mask)
    return masked

    

def selectRegions(image  , text , flag):
    global refPt , cropped
    clone = copy.deepcopy(image)
    while True:
      key = cv2.waitKey(1) & 0xFF
      # display the image and wait for a keypress
      cv2.putText(image, text ,  (60,30), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)
      cv2.putText(image, "Press 'r' key to reset everything.",  (60,70), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)
      cv2.putText(image, "Press 'd' key if the region selection is done.",  (60,110), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)

      for pt in range(len(refPt)-1):
        pt1 , pt2 = refPt[pt] , refPt[pt+1]
        cv2.line(image, (pt1[0],pt1[1]), (pt2[0],pt2[1]), [0,255,255], 3)      

      cv2.imshow("ROI", image)
      if key == ord("r"):
        image = copy.deepcopy(clone)
        refPt = []
      elif key == ord("d"):
        if len(refPt) > 2:
          cropped = flag
          vertices = np.array(refPt, np.int32)
          image = roi(clone, [vertices])
          cv2.imshow("ROI", image)
          cap.set(1,start_frame)
          return 
      elif key == ord('q'):
        return 1



# def visualize(output_dict,image_np,height,width):
#   class_ids = []
#   confidences = []
#   boxes = []
#   num = output_dict['num_detections']
#   for ind in range(num):
#     scr = output_dict['detection_scores'][ind]
#     classId = output_dict['detection_classes'][ind] 
#     box = output_dict['detection_boxes'][ind]

#     ymin, xmin, ymax, xmax = box
#     confidences.append(float(scr))
#     class_ids.append(classId)
#     boxes.append([int(xmin*width) , int(ymin*height) , int((xmax-xmin)*width) , int((ymax-ymin)*height)])

#   indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#   # if len(boxes) != len(indexes):
#   #   print(indexes,boxes , confidences,class_ids)
#   for j in indexes:
#     i = j[0]
#     x, y, w, h = boxes[i]
#     label = category_index[class_ids[i]]['name']
#     color = colors[i]
#     cv2.rectangle(image_np, (x, y), (x + w, y + h), color, 3)
#     cv2.putText(image_np, label, (x, y - 5), font, 3, color, 3)
  # return image_np






def show_inference(model, image_path):
  global number , prev_frame
  global signalCounter , flagSignal
  global crash_count_frames 
  global flagPerson , areaPerson
  image_np = np.array(image_path)
  lane_image = copy.deepcopy(image_np)
  gray_image = cv2.cvtColor(copy.deepcopy(image_np), cv2.COLOR_BGR2GRAY)

  mask = 255*np.ones_like(image_np)
  vertices = np.array(dashPointer, np.int32)
  cv2.fillPoly(mask, [vertices], [0,0,0])
  # cv2.imshow("check dash",mask)
  image_np = cv2.bitwise_and(image_np, mask)
  # cv2.imshow("dash removed",image_np)

  height,width,channel = image_np.shape
  image = np.asarray(image_np)
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

  confidencesCars , boxesCars = [] , []
  confidencesLights , boxesLights = [] , []
  confidencesPersons , boxesPersons = [] , []

  num = output_dict['num_detections']
  for ind in range(num):
    classId = output_dict['detection_classes'][ind] 
    if classId==3 or classId==6 or classId==8:
      scr = output_dict['detection_scores'][ind]
      box = output_dict['detection_boxes'][ind]
      ymin, xmin, ymax, xmax = box
      w = (xmax - xmin) * width
      h = (ymax - ymin) * height
      if (w*h >=800):
        boxesCars.append([int(xmin*width) , int(ymin*height) , int(w) , int(h)])
        confidencesCars.append(float(scr))
    elif classId==10:
      scr = output_dict['detection_scores'][ind]
      box = output_dict['detection_boxes'][ind]
      ymin, xmin, ymax, xmax = box
      confidencesLights.append(float(scr))
      boxesLights.append([int(xmin*width) , int(ymin*height) , int((xmax-xmin)*width) , int((ymax-ymin)*height)])
    elif classId==1:
      scr = output_dict['detection_scores'][ind]
      box = output_dict['detection_boxes'][ind]
      ymin, xmin, ymax, xmax = box
      confidencesPersons.append(float(scr))
      boxesPersons.append([int(xmin*width) , int(ymin*height) , int((xmax-xmin)*width) , int((ymax-ymin)*height)])


  indexesLights = cv2.dnn.NMSBoxes(boxesLights, confidencesLights, 0.5, 0.4)
  indexesCars = cv2.dnn.NMSBoxes(boxesCars, confidencesCars, 0.5, 0.4)
  indexesPersons = cv2.dnn.NMSBoxes(boxesPersons, confidencesPersons, 0.5, 0.4)


  image_np = signalDetection_utils.signalDetection(indexesLights , boxesLights , image_np)
  image_np = tracking_utils.tracking(indexesCars , boxesCars , image_np)
  image_np = estimate_collide_utils.estimate_collide(indexesCars , boxesCars , image_np)
  image_np = estimate_stepping_utils.estimate_stepping(indexesPersons , boxesPersons , image_np)


  canny_image =  cv2.Canny(gray_image, threshold1 = 200, threshold2=300)
  canny_image = cv2.GaussianBlur(canny_image,(3,3),0)
  vertices = np.array(lanePointer, np.int32)
  mask = np.zeros_like(canny_image)
  cv2.fillPoly(mask, [vertices], [255,255,255])
  canny_image = cv2.bitwise_and(canny_image, mask)
  cv2.imshow("canny_image",canny_image)

  lines = cv2.HoughLinesP(canny_image, 1, np.pi/180, 180, np.array([]), minLineLength = 5, maxLineGap = 5)
  lane_detection_utils.draw_lines(image_np , lines , lane_image)
  # lane_detection_utils.all_lines(image_np , lines , lane_image)


  cv2.imshow("finally", image_np)








refPt = []
cropped = 0
cap=cv2.VideoCapture('../videos/d.mp4')
start_frame =90*24
cap.set(1,start_frame)
_ , image = cap.read()
cv2.namedWindow("ROI")
cv2.setMouseCallback("ROI", click_and_crop)
ctt = 0
Quit = 0
while True:
  key = cv2.waitKey(1) & 0xFF
  if cropped == 0:
    Quit = selectRegions(copy.deepcopy(image)  , "Click points to select your vehicle dash." , 1)
    dashPointer = refPt
    refPt = []
    print("For dash: ",dashPointer)
  elif cropped == 1:
    Quit = selectRegions(copy.deepcopy(image)  , "Click points to select bird's eye view." , 2)
    lanePointer = refPt
    print("For lanes: ",lanePointer)
    fps = FPS().start()
  else:
    _,frame = cap.read()
    if _ == False:
      break
    ctt = ctt + 1
    # print(ctt)
    show_inference(detection_model, frame)

    cv2.imshow("original",frame)
    fps.update()
    if key == ord('q'):
      fps.stop()
      print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
      print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
      break
  if Quit:
    break

cv2.destroyAllWindows()
cap.release()
