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

colors = np.random.uniform(0, 255, size=(100, 3))
font = cv2.FONT_HERSHEY_SIMPLEX



startRedLower = (0 , 130 , 50)
startRedUpper = (13 , 255, 255)
endRedLower = (150 , 130 , 50)
endRedUpper = (180 , 255 , 255)



def signalDetection(indexesLights , boxesLights , image_np , signalCounter , flagSignal):
  maskRed = np.zeros_like(image_np)
  lighsImg = copy.deepcopy(image_np)
  # trafficLights = []
  areas = []
  boxes = []
  for j in indexesLights:
    i = j[0]
    x, y, w, h = boxesLights[i]
    label = (w * h)
    if label < 450:
      label = " "
    else:
      cv2.rectangle(image_np, (x, y), (x + w, y + h), (255,255,0), 2)
      # cv2.putText(image_np, str(label), (x, y - 5), font, 3, (255,255,0), 2)
    # trafficLights.append([x , y , w , h , str(label)])
    crop = image_np[y:y+h , x:x+w , :]
    maskRed[y:y+h , x:x+w , :] = crop

    cv2.rectangle(lighsImg, (x, y), (x + w, y + h), (255,255,0), 2)
    cv2.putText(lighsImg, str(label), (x, y - 5), font, 1.2, (255,255,0), 2)

  cv2.imshow("lights bounding boxes" , lighsImg)
  cv2.imshow("box-thresholding lights",maskRed)

  blurred = cv2.GaussianBlur(maskRed, (11, 11), 0)
  hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
  mask1 = cv2.inRange(hsv, startRedLower, startRedUpper)
  mask2 = cv2.inRange(hsv, endRedLower, endRedUpper)
  maskRed = mask1 + mask2
  maskRed = cv2.erode(maskRed, None, iterations=2)
  maskRed = cv2.dilate(maskRed, None, iterations=2)
  cv2.imshow("red contours",maskRed)

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
      if radius >= 3:
          hull.append(chull)
          redcircles.append([radius , cX , cY]) 
          flag = 1
  if flag == 1:
    flagSignal.append(1)
  else:
    flagSignal.append(0)
  
  # cv2.putText(image_np, str(flagSignal), (30,130), font, 1.2, (0,0,255), 2,cv2.LINE_AA)                 # array of 20 integers in flagSignal

  if sum(flagSignal) > 5:
    cv2.putText(image_np, "Hey!! traffic signal is red", (340,160), font, 1.2, (0,0,255), 2,cv2.LINE_AA)
    signalCounter = 1
  else:
    signalCounter = signalCounter - 1
  if -16 < signalCounter <= 0:
    cv2.putText(image_np, "You can drive now", (400,160), font, 1.2, (0,255,255), 2,cv2.LINE_AA)


  # draw contours and hull points
  for i in range(len(hull)):
      color_contours = (0, 255, 0) # green - color for contours
      color_hull = (0, 255, 255) # blue - color for convex hull
      # draw ith contour
      # cv2.putText(image_np, str(redcircles[i][0]), (redcircles[i][1] - 5, redcircles[i][2] - 5), font, 1.2, (255,255,255), 2)
      cv2.drawContours(image_np, contours, i, color_contours, 1, 8, hierarchy)
      cv2.drawContours(image_np, hull, i, color_hull, 2, 8)  
  return image_np , signalCounter , flagSignal 








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

# n p s t
