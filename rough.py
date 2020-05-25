import numpy as np
import cv2
import imutils
import time
from sklearn.metrics import pairwise
from imutils.video import FPS
import copy
import os
import sys
import pathlib
from collections import defaultdict

# import tensorflow as tf

from utils import tracking_utils 
from utils import signalDetection_utils 
from utils import estimate_collide_utils
from utils import estimate_stepping_utils
from utils import lane_detection_utils
from utils import break_light_utils

from utils import ops as utils_ops
from utils import label_map_util

# utils_ops.tf = tf.compat.v1
# tf.gfile = tf.io.gfile
PATH_TO_LABELS = '../bigdata/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



# print(category_index)
colors = np.random.uniform(0, 255, size=(len(category_index), 3))
font = cv2.FONT_HERSHEY_SIMPLEX




def click_and_crop(event, x, y, flags, param):
	global refPt
	# if the left mouse button was clicked, record the starting (x, y) coordinates
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt.append([x, y])

	for pt in range(len(refPt)-1):
		pt1 , pt2 = refPt[pt] , refPt[pt+1]
		cv2.line(image, (pt1[0],pt1[1]), (pt2[0],pt2[1]), [0,255,255], 3)      


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, [255,255,255])
    # cv2.imshow("mask",mask)
    masked = cv2.bitwise_and(img, mask)
    return masked

    







blackLower = (0 , 0 , 0)
blackUpper = (180 , 255 , 35)



def confirm_day_or_night(frame , flag_night_counter):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blackLower , blackUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask , None, iterations=2)
    # cv2.imshow('black',imutils.resize(mask,width=250))
    cv2.imshow('frame',frame)
    pixel_ct = 0
    pixel_len = 0
    for i in mask:
      pixel_ct = pixel_ct + np.sum(i==0)
      pixel_len = pixel_len + len(i)
    ratio = pixel_ct / pixel_len
    print("ratio = ",ratio)
    if ratio < 0.68:
        flag_night_counter = flag_night_counter + 1
        return flag_night_counter
    else:
        flag_night_counter = flag_night_counter - 1 
        return flag_night_counter






def selectRegions(image  , text , flag):
    global refPt , cropped
    clone = copy.deepcopy(image)
    while True:
      key = cv2.waitKey(1) & 0xFF
      # display the image and wait for a keypress
      cv2.putText(image, text ,  (60,30), font , 2, [0,255,255], 3)
      cv2.putText(image, "Press 'r' key to reset everything.",  (60,70), font , 2, [0,255,255], 3)
      cv2.putText(image, "Press 'd' key if the region selection is done.",  (60,110), font , 2, [0,255,255], 3)

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





refPt = []
cropped = 0
cap=cv2.VideoCapture('../videos/a.mp4')
start_frame =292*25
cap.set(1,start_frame)
_ , image = cap.read()
cv2.namedWindow("ROI")
cv2.setMouseCallback("ROI", click_and_crop)
ctt = 0
Quit = 0
flag_night_counter = 0
flagConfirm = 0
while True:
	(grabbed, frame) = cap.read()
	height,width,channel = frame.shape
	if flagConfirm == 0:
		for z in range(10):
			flag_night_counter = confirm_day_or_night(frame , flag_night_counter)
		cv2.destroyWindow("frame")
		print("flag_night_counter = ",flag_night_counter)
		cap.set(1 , start_frame)
		flagConfirm = 1
	else:
		if flag_night_counter > 4:
			key = cv2.waitKey(1) & 0xFF
			if cropped == 0:
				Quit = selectRegions(copy.deepcopy(image)  , "Click points to select your vehicle dash." , 1)
				dashPointer = refPt
				refPt = []
				print("For dash: ",dashPointer)
				fps = FPS().start()
			else:
				_,frame = cap.read()
				if _ == False:
					fps.stop()
					print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
					print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
					break
				ctt = ctt + 1
				# print(ctt)
				break_light_utils.break_light(dashPointer , frame)

				# cv2.imshow("original",frame)
				fps.update()
				if key == ord('q'):
					fps.stop()
					print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
					print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
					break
			if Quit:
				break

		else:                                                                                                   # DAY TIME
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
					fps.stop()
					print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
					print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
					break
				ctt = ctt + 1
				# print(ctt)
				cv2.putText(frame,"DAY",(width - 200 ,50), font, 2,(167,133,0),2,cv2.LINE_AA)

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
