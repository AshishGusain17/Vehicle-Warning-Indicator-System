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



# def all_lines(lanePointer , lane_image , image_np):
#     height , width , channels= image_np.shape
#     gray_image = cv2.cvtColor(lane_image , cv2.COLOR_BGR2GRAY)
#     canny_image =  cv2.Canny(gray_image, threshold1 = 200, threshold2=300)
#     canny_image = cv2.GaussianBlur(canny_image,(3,3),0)
#     vertices = np.array(lanePointer, np.int32)
#     mask = np.zeros_like(canny_image)
#     cv2.fillPoly(mask, [vertices], [255,255,255])
#     canny_image = cv2.bitwise_and(canny_image, mask)
#     cv2.imshow("canny_image",canny_image)
#     try:
#         for line in lines:
#             coords = line[0]
#             cv2.line(lane_image, (coords[0],coords[1]), (coords[2],coords[3]), [0,255,255], 3)         # yellow color vertical
#     except:
#         pass

#     cv2.imshow("lane_image",lane_image)



def draw_lines(lanePointer , dashPointer , lane_image , image_np , flagLanes):
    height , width , channels= image_np.shape
    gray_image = cv2.cvtColor(lane_image , cv2.COLOR_BGR2GRAY)
    canny_image =  cv2.Canny(gray_image, threshold1 = 100, threshold2=100)
    # cv2.imshow("entire canny",canny_image)
    canny_image = cv2.GaussianBlur(canny_image,(3,3),0)

    mask = np.zeros_like(canny_image)
    vertices = np.array(lanePointer, np.int32)
    cv2.fillPoly(mask, [vertices], [255,255,255])

    # cv2.imshow("mask",mask)
    vertices = np.array(dashPointer, np.int32)
    cv2.fillPoly(mask, [vertices], [0,0,0])

    canny_image = cv2.bitwise_and(canny_image, mask)
    cv2.imshow("canny with mask",canny_image)

    # cv2.putText(lane_image, str(flagLanes), (30,130), font, 1.2, (0,0,255), 2,cv2.LINE_AA)                 # array of 20 integers in flagLanes

    lines = cv2.HoughLinesP(canny_image, 1, np.pi/180, 180, np.array([]), minLineLength = 15, maxLineGap = 15)
    try:
        flagCounter = 0
        if len(lines):
            flagLanes.pop(0)
        for line in lines:
            coords = line[0]
            x1 , y1 , x2 , y2 = coords[0] , coords[1] , coords[2] , coords[3]
            if x2 == x1:
                cv2.line(lane_image, (x1 , y1), (x2 , y2), [0,255,255], 3)                          # yellow color vertical
                just_to_pass = 0
            else:
                slope=(y1 - y2)/(x2 - x1)
                if -0.3 < slope < 0.3:
                    cv2.line(lane_image, (x1 , y1), (x2 , y2), [255,0,0], 2)                        # blue color horizontal
                    just_to_pass = 0
                elif slope < 0:
                    if width//2 > max(x1 , x2):
                        slope=str(slope)[:5]
                        cv2.putText(lane_image, str(slope),  (x1 , y1), font, 1, [122,32,12], 2)
                        cv2.line(lane_image, (x1 , y1), (x2 , y2), [0,0,0], 2)                      # black color vertical
                        flagCounter = 1
                    else:
                        slope=str(slope)[:5]
                        cv2.putText(lane_image, str(slope),  (x1 , y1), font, 1, [122,32,12], 2)
                        cv2.line(lane_image, (x1 , y1), (x2 , y2), [0,255,255], 2)                   # yellow color vertical

                elif slope > 0:
                    if width//2 < min(x1 , x2):
                        slope=str(slope)[:5]
                        cv2.putText(lane_image, str(slope),  (x1 , y1), font, 1, [122,32,12], 2)
                        cv2.line(lane_image, (x1 , y1), (x2 , y2), [0,0,0], 2)                       # black color vertical
                        flagCounter = 1
                    else:
                        slope=str(slope)[:5]
                        cv2.putText(lane_image, str(slope),  (x1 , y1), font, 1, [122,32,12], 2)
                        cv2.line(lane_image, (x1 , y1), (x2 , y2), [0,255,255], 2)                   # yellow color vertical
        if flagCounter == 1:
            flagLanes.append(1)
        else:
            flagLanes.append(0)
        if sum(flagLanes) > 12:
            cv2.putText(image_np, "Get back to your lane" ,  (370,80), font , 1.2, (0,255,0), 2,cv2.LINE_AA)


    except:
        pass
    cv2.imshow("lane_image",lane_image)
    return image_np , flagLanes








# lanes   r
# a   451(lanes showing good)
# b   115(warning shows good )
# d   0
# d   81






