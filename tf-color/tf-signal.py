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
import copy


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





# original
# startRedLower = (0 , 180 , 50)
# startRedUpper = (15 , 255, 255)
# endRedLower = (165 , 180 , 50)
# endRedUpper = (180 , 255 , 255)

# a.mp4 659(changed from above to this)     b.mp4(147) not-working 
# startRedLower = (0 , 130 , 50)
# startRedUpper = (15 , 255, 255)
# endRedLower = (165 , 130 , 50)
# endRedUpper = (180 , 255 , 255)

# b.mp4  147 (changed from above to this)
# startRedLower = (0 , 130 , 50)
# startRedUpper = (13 , 255, 255)
# endRedLower = (167 , 130 , 50)
# endRedUpper = (180 , 255 , 255)

# d.mp4  164(changed from above to this)
# startRedLower = (0 , 130 , 50)
# startRedUpper = (13 , 255, 255)
# endRedLower = (150 , 130 , 50)
# endRedUpper = (180 , 255 , 255)


  

colors = np.random.uniform(0, 255, size=(len(category_index), 3))
font = cv2.FONT_HERSHEY_PLAIN

startRedLower = (0 , 130 , 50)
startRedUpper = (13 , 255, 255)
endRedLower = (150 , 130 , 50)
endRedUpper = (180 , 255 , 255)
signalCounter = -99999
flagSignal = [0] * 10
def signalDetection(image_np , indexesLights , boxesLights):
  global signalCounter , flagSignal
  maskRed = np.zeros_like(image_np)
  fr = copy.deepcopy(image_np)
  trafficLights = []
  areas = []
  boxes = []
  for j in indexesLights:
    i = j[0]
    x, y, w, h = boxesLights[i]
    label = (w * h)
    if label < 450:
      label = "less"
    else:
      cv2.rectangle(image_np, (x, y), (x + w, y + h), (255,255,0), 2)
      cv2.putText(image_np, str(label), (x, y - 5), font, 3, (255,255,0), 2)
    trafficLights.append([x , y , w , h , str(label)])
    crop = image_np[y:y+h , x:x+w , :]
    maskRed[y:y+h , x:x+w , :] = crop
    color = colors[i]
    cv2.rectangle(fr, (x, y), (x + w, y + h), (255,255,0), 2)
    cv2.putText(fr, str(label), (x, y - 5), font, 3, (255,255,0), 2)

  cv2.imshow("light boxes" , fr)
  cv2.imshow("crop",maskRed)
  
  blurred = cv2.GaussianBlur(maskRed, (11, 11), 0)
  hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
  mask1 = cv2.inRange(hsv, startRedLower, startRedUpper)
  mask2 = cv2.inRange(hsv, endRedLower, endRedUpper)
  maskRed = mask1 + mask2
  maskRed = cv2.erode(maskRed, None, iterations=2)
  maskRed = cv2.dilate(maskRed, None, iterations=2)

  (_, contours , hierarchy) = cv2.findContours(maskRed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  hull = []
  redcircles = []
  flagSignal.pop(0) 
  flag = 0
  for i in range(len(contours)):
      chull = cv2.convexHull(contours[i], False)
      extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
      extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
      extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
      extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])
      cX = int((extreme_left[0] + extreme_right[0]) / 2)
      cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
      distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
      radius = int(distance[distance.argmax()])
      if radius >= 4:
          hull.append(chull)
          redcircles.append([radius , cX , cY]) 
          flag = 1
  if flag == 1:
    flagSignal.append(1)
  else:
    flagSignal.append(0)
  if sum(flagSignal) > 5:
    cv2.putText(image_np, "Hey !! traffic signal is red", (30,30), font, 1.2, (0,255,255), 2,cv2.LINE_AA)
    signalCounter = 8
  else:
    signalCounter = signalCounter - 1
  if -20 < signalCounter <= 0:
    cv2.putText(image_np, "You can move now", (30,30), font, 1.2, (0,255,255), 2,cv2.LINE_AA)

  # print(len(hull))

  # draw contours and hull points
  for i in range(len(hull)):
      color_contours = (0, 255, 0) # green - color for contours
      color_hull = (0, 255, 255) # blue - color for convex hull
      # draw ith contour
      cv2.putText(image_np, str(redcircles[i][0]), (redcircles[i][1] - 5, redcircles[i][2] - 5), font, 2, (255,255,255), 2)
      cv2.drawContours(image_np, contours, i, color_contours, 1, 8, hierarchy)
      cv2.drawContours(image_np, hull, i, color_hull, 2, 8)  
  return image_np


def visualize(output_dict,image_np,height,width):
  class_ids , confidences , boxes = [] , [] , []
  boxesLights , confidencesLights = [] , []
  num = output_dict['num_detections']
  for ind in range(num):
    scr = output_dict['detection_scores'][ind]
    classId = output_dict['detection_classes'][ind] 
    box = output_dict['detection_boxes'][ind]
    if classId ==10:
      ymin, xmin, ymax, xmax = box
      confidencesLights.append(float(scr))
      boxesLights.append([int(xmin*width) , int(ymin*height) , int((xmax-xmin)*width) , int((ymax-ymin)*height)])
    elif  classId==2 or classId==3 or classId==4 or classId==6 or classId==8:
      pass

  indexesLights = cv2.dnn.NMSBoxes(boxesLights, confidencesLights, 0.5, 0.4)
  maskRed = signalDetection(image_np , indexesLights , boxesLights)
  return maskRed



def show_inference(model, image_path,ctt):
  image_np = np.array(image_path)
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

  maskRed = visualize(output_dict,image_np,height,width)

  # cv2.imshow("traffic light", image_np)
  cv2.imshow("red",maskRed)




# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('../videos/i.mp4')
cap.set(1,537*24)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out1 = cv2.VideoWriter('i.avi', fourcc, 3.0, (int(cap.get(3)),int(cap.get(4))))

fps = FPS().start()

ctt = 0
while True:
    (grabbed, frame) = cap.read()
    if grabbed != True:
      break
    # print(ctt)
    ctt = ctt + 1

    show_inference(detection_model, frame,ctt)
    

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



# size of traffic lights
# size of radius
# number of hulls inside tf light
# clear to go

# a.mp4   210*25(red)    238*25(green)   273*25(red)    291*25(red-green-orange)   358*25(red)      659*25(red)   903*25(red)   937(red)
# b.mp4   6*24(green)     147*24(yellow red doubt )   339*24
# c.mp4   90*24   342(no-light)    441(red)    525
# d.mp4   0(green)    164(red)    221    233     379(green-red)-d   467
# e.mp4   44(red-green)    74(red)    416
# f.mp4   0(red)   128(red)    178    311(green)
# g.mp4   110    141    209   285
# h.mp4   139   401
# i.mp4   27   231(red-green)-d    252    378  537


