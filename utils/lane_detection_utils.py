import numpy as np
import os
import sys
import tensorflow as tf
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



def all_lines(lanePointer , lane_image , image_np):
    height , width , channels= image_np.shape
    gray_image = cv2.cvtColor(lane_image , cv2.COLOR_BGR2GRAY)
    canny_image =  cv2.Canny(gray_image, threshold1 = 200, threshold2=300)
    canny_image = cv2.GaussianBlur(canny_image,(3,3),0)
    vertices = np.array(lanePointer, np.int32)
    mask = np.zeros_like(canny_image)
    cv2.fillPoly(mask, [vertices], [255,255,255])
    canny_image = cv2.bitwise_and(canny_image, mask)
    cv2.imshow("canny_image",canny_image)
    try:
        for line in lines:
            coords = line[0]
            cv2.line(lane_image, (coords[0],coords[1]), (coords[2],coords[3]), [0,255,255], 3)         # yellow color vertical
    except:
        pass
    # print("exception")
    cv2.imshow("lane_image",lane_image)




def draw_lines(lanePointer , lane_image , image_np):
    height , width , channels= image_np.shape
    gray_image = cv2.cvtColor(lane_image , cv2.COLOR_BGR2GRAY)
    canny_image =  cv2.Canny(gray_image, threshold1 = 200, threshold2=300)
    canny_image = cv2.GaussianBlur(canny_image,(3,3),0)
    vertices = np.array(lanePointer, np.int32)
    mask = np.zeros_like(canny_image)
    cv2.fillPoly(mask, [vertices], [255,255,255])
    canny_image = cv2.bitwise_and(canny_image, mask)
    cv2.imshow("canny_image",canny_image)

    lines = cv2.HoughLinesP(canny_image, 1, np.pi/180, 180, np.array([]), minLineLength = 5, maxLineGap = 5)
    try:
        for line in lines:
            coords = line[0]
            flag = 0 
            if coords[2]==coords[0]:
                # print("infinite slope")
                cv2.line(lane_image, (coords[0],coords[1]), (coords[2],coords[3]), [0,255,255], 3)         # yellow color vertical
            else:
                slope=(coords[1]-coords[3])/(coords[2]-coords[0])
                if -0.3 < slope < 0.3:
                    cv2.line(lane_image, (coords[0],coords[1]), (coords[2],coords[3]), [255,0,0], 2)           # blue color horizontal
                elif slope < 0:
                    # if (coords[0] + coords[2])/2 < width//2 < max([coords[0],coords[2]]):
                    if min([coords[0] , coords[2]]) < width//2 < max([coords[0],coords[2]]):
                        flag=1 
                        slope=str(slope)[:5]
                        cv2.putText(lane_image, str(slope),  (coords[0],coords[1]), font, 3, [122,32,12], 2)
                        cv2.line(lane_image, (coords[0],coords[1]), (coords[2],coords[3]), [0,0,0], 2)         # black color vertical
                        cv2.putText(lane_image, "Get back to your lane" ,  (370,80), font , 1.2, (0,255,0), 2,cv2.LINE_AA)

                elif slope > 0:
                    # if (coords[0] + coords[2])/2 > width//2 > min([coords[0],coords[2]]):
                    if max([coords[0],coords[2]]) > width//2 > min([coords[0],coords[2]]):
                        flag=1 
                        slope=str(slope)[:5]
                        cv2.putText(lane_image, str(slope),  (coords[0],coords[1]), font, 3, [122,32,12], 2)
                        cv2.line(lane_image, (coords[0],coords[1]), (coords[2],coords[3]), [0,0,0], 2)         # black color vertical
                        cv2.putText(lane_image, "Get back to your lane" ,  (370,80), font , 1.2, (0,255,0), 2,cv2.LINE_AA)

                    if flag == 0:
                        slope=str(slope)[:5]
                        cv2.putText(lane_image, str(slope),  (coords[0],coords[1]), font, 3, [122,32,12], 2)
                        cv2.line(lane_image, (coords[0],coords[1]), (coords[2],coords[3]), [0,255,255], 2)         # yellow color vertical


    except:
        pass
    cv2.imshow("lane_image",lane_image)
    return image_np 


    
