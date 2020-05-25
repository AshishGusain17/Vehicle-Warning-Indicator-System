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


def estimate_collide(indexesCars , boxesCars , image_np , crash_count_frames):
	height , width , channel = image_np.shape
	vehicle_crash = 0
	max_curr_obj_area = 0
	centerX = centerY = 0
	details = [0 , 0 , 0 , 0]
	for j in indexesCars:
		i = j[0]
		xmin, ymin, w, h = boxesCars[i]
		obj_area = w * h
		if obj_area > max_curr_obj_area:
			max_curr_obj_area = obj_area
			details = [ymin, xmin, ymin+h, xmin+w]


	cv2.putText(image_np,str(max_curr_obj_area) ,(50,250), font, 1.2,(255,255,0),2,cv2.LINE_AA)

	centerX , centerY = (details[1] + details[3])/(2*width) , (details[0] + details[2])/(2*height)
	if max_curr_obj_area>40000:
		# if (centerX < 0.2 and details[2] > 0.9) or (0.3 <= centerX <= 0.7) or (centerX > 0.8 and details[2] > 0.9):
		# print(centerX , 12)
		if 0.27 <= centerX <= 0.73:
			vehicle_crash = 1
			crash_count_frames = 10


	if vehicle_crash == 0:
		crash_count_frames = crash_count_frames - 1

	elif crash_count_frames > 0:
		if max_curr_obj_area <= 70000:
			cv2.putText(image_np,"YOU ARE GETTING CLOSER" ,(50,50), font, 1.2,(0,255,255),2,cv2.LINE_AA)
		elif max_curr_obj_area > 70000:
			cv2.putText(image_np,"DON'T COLLIDE !!!" ,(50,50), font, 1.2,(0,0,255),2,cv2.LINE_AA)

	return image_np , crash_count_frames



# a.mp4(25)   56    74  110
# b.mp4(24)  5  270   292  368    509
# c.mp4(24)   0  111    166  189(many cars, but not in range)   290    494
# d.mp4  2
