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


startRedLower = (0 , 180 , 90)
startRedUpper = (10 , 255, 255)
endRedLower = (170 , 180 , 90)
endRedUpper = (180 , 255 , 255)


def break_light(dashPointer , image_np):
	image_np = copy.deepcopy(image_np)
	height , width , channels = image_np.shape

	mask = 255*np.ones_like(image_np)
	vertices = np.array(dashPointer, np.int32)
	cv2.fillPoly(mask, [vertices], [0,0,0])
	mask = cv2.bitwise_and(image_np, mask)
	cv2.imshow("dash removed frame",mask)


	blurred = cv2.GaussianBlur(mask, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	mask1 = cv2.inRange(hsv, startRedLower, startRedUpper)
	mask2 = cv2.inRange(hsv, endRedLower, endRedUpper)
	maskRed = mask1 + mask2
	maskRed = cv2.erode(maskRed, None, iterations=2)
	maskRed = cv2.dilate(maskRed, None, iterations=2)


	(_, contours , hierarchy) = cv2.findContours(maskRed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	hull = []
	indHull = []
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
		hull.append(chull)
		if radius >15:
			# cv2.circle(image_np, (int(cX), int(cY)), int(radius),(167,133,0), 2)
			indHull.append(i)
			cv2.putText(image_np,"Let me show you brake-lights radiations patches.",(170 ,80),  font, 1.2 , (0, 255, 255) , 2 , cv2.LINE_AA)
			cv2.putText(image_np,"Apply brakes accordingly.",                       (390 ,120), font, 1.2 , (0, 255, 255) , 2 , cv2.LINE_AA)


	# print("length = ",len(hull))

	for i in indHull:
		color_contours = (0, 255, 0) # green - color for contours
		color = (0, 255, 255) # blue - color for convex hull
		# draw ith contour
		cv2.drawContours(image_np, contours, i, color_contours, 1, 8, hierarchy)
		# draw ith convex hull object
		cv2.drawContours(image_np, hull, i, color, 2, 8) 

	cv2.putText(image_np,"NIGHT",(50 ,90), font, 2,(167,133,0),2,cv2.LINE_AA)                   # NIGHT TIME
	cv2.imshow("finally" , image_np)




# a.mp4(25fps)   25*25    292*25    
# c.mp4(24fps) 133*24 + 6
# d.mp4(24fps) 195*24   419*24
# e.mp4(24fps)   283    338
# f.mp4(24fps) 36  223
# j.mp4(30fps)    false results
# l.mp4(30fps)   54
# i.mp4(24)    267
# g.mp4(24)    394