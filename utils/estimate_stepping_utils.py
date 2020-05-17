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
font = cv2.FONT_HERSHEY_PLAIN



# flagPerson and areaPerson are pedestrian crash
flagPerson = 0
areaPerson = 0
areaDetails = []
def estimate_stepping(indexesPersons , boxesPersons , image_np):
	global flagPerson , areaPerson , areaDetails
	pedes_present = 0
	details = []
	for j in indexesPersons:
		i = j[0]
		xmin, ymin, w, h = boxesPersons[i]
		curr_area = w * h
		if curr_area > 25000:
			areaPerson = curr_area
			pedes_present = 1
			flagPerson = 6
			details.append([xmin, ymin, w, h])

	if pedes_present == 0:
		flagPerson = flagPerson - 1
	else:
		areaPerson = 0
		for box in details:
			xmin, ymin, w, h = box
			boxArea = w * h
			cv2.rectangle(image_np, (xmin, ymin), (xmin + w, ymin + h), (0,0,0), 3)
			cv2.putText(image_np, str(boxArea),  (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 2, [0,0,0], 3)
			if boxArea > areaPerson:
				areaPerson = boxArea
		areaDetails = details

	if flagPerson > 0:
		for box in areaDetails:
			xmin, ymin, w, h = box
			cv2.rectangle(image_np, (xmin, ymin), (xmin + w, ymin + h), (0,0,0), 3)
			cv2.putText(image_np, str(areaPerson),  (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, 2, [0,0,0], 3)

		if areaPerson > 15000:
		  cv2.putText(image_np,"STOP IT !!! DON'T HIT THE PERSON " + str(areaPerson),(50,50), font, 3,(255,255,0),2,cv2.LINE_AA)
		else:
		  cv2.putText(image_np,"Drive slowly, person is ahead " + str(areaPerson),(50,50), font, 3,(255,255,0),2,cv2.LINE_AA)

	return image_np









# a.mp4   100*25      803*25(inside cars)  819*25     842*25      913*25