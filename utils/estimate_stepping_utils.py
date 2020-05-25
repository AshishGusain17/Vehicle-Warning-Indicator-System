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




def estimate_stepping(indexesPersons , boxesPersons , image_np , flagPerson , areaPerson , areaDetails):
	pedes_present = 0
	details = []
	for j in indexesPersons:
		i = j[0]
		xmin, ymin, w, h = boxesPersons[i]
		curr_area = w * h
		if curr_area > 9000:
			areaPerson = curr_area
			pedes_present = 1
			flagPerson = 5
			details.append([xmin, ymin, w, h])

	if pedes_present == 0:
		flagPerson = flagPerson - 1
	else:
		areaPerson = 0
		for box in details:
			xmin, ymin, w, h = box
			boxArea = w * h
			cv2.rectangle(image_np, (xmin, ymin), (xmin + w, ymin + h), (0,0,0), 3)
			cv2.putText(image_np, str(boxArea),  (xmin, ymin), font , 1.2, [0,0,0], 2)
			if boxArea > areaPerson:
				areaPerson = boxArea
		areaDetails = details

	if flagPerson > 0:
		for box in areaDetails:
			xmin, ymin, w, h = box
			cv2.rectangle(image_np, (xmin, ymin), (xmin + w, ymin + h), (0,0,0), 3)
			cv2.putText(image_np, str(areaPerson),  (xmin, ymin), font , 1.2, [0,0,0], 2)

		if areaPerson > 15000:
			cv2.putText(image_np,"STOP IT !!! DON'T HIT THE PERSON " ,(270,120), font, 1.2,(0,0,255),2,cv2.LINE_AA)
		else:
			cv2.putText(image_np,"Drive slowly, people are around "    ,(290,120), font, 1.2,(0,255,255),2,cv2.LINE_AA)

	return image_np , flagPerson , areaPerson , areaDetails 








# a.mp4   100*25      803*25(inside cars)  819*25     842*25      913*25
# k.mp4(30)   22(don't use, night)   83(don't night)
# m.mp4(24)         6    25
# o.mp4(30)    2
# p.mp4(30)      3
# q.mp4(30)    20   0