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



def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # print(interArea, float(boxAArea + boxBArea - interArea))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou




def get_dict(dashPointer , model , image_np):
    mask = 255*np.ones_like(image_np)
    vertices = np.array(dashPointer, np.int32)
    cv2.fillPoly(mask, [vertices], [0,0,0])
    image_np = cv2.bitwise_and(image_np, mask)

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
    return output_dict



def findBoxes(image_np , output_dict):
    height,width,channel = image_np.shape
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
                confidencesCars.append(float(scr))
                boxesCars.append([int(xmin*width) , int(ymin*height) , int(w) , int(h)])
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

    return confidencesCars , boxesCars , confidencesLights , boxesLights , confidencesPersons , boxesPersons
