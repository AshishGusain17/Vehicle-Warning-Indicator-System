import numpy as np
import cv2
import imutils
import time
from imutils.video import FPS
from sklearn.metrics import pairwise
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import pathlib
from collections import defaultdict
import copy



font = cv2.FONT_HERSHEY_SIMPLEX

startRedLower = (0 , 180 , 90)
startRedUpper = (10 , 255, 255)
endRedLower = (170 , 180 , 90)
endRedUpper = (180 , 255 , 255)

blackLower = (0 , 0 , 0)
blackUpper = (180 , 255 , 35)



def confirm_day_or_night(frame , flag_night_counter):
    mask = cv2.inRange(hsv, blackLower , blackUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask , None, iterations=2)
    cv2.imshow('black',imutils.resize(mask,width=250))
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



cap=cv2.VideoCapture('../videos/l.mp4')
set_pos=54*30

cap.set(1,set_pos)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out1 = cv2.VideoWriter('brakes.avi', fourcc, 25, (1280,720))
fps = FPS().start()

ct=0
flag_night_counter = 0
initial_flag = 0
while True:
    (grabbed, frame) = cap.read()
    frame=imutils.resize(frame, width=1280)
    height,width,channel = frame.shape
    # print(frame.shape)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    if initial_flag == 0:
        ct = ct + 1
        flag_night_counter = confirm_day_or_night(frame , flag_night_counter)
        if ct == 10:
            print("flag_night_counter = ",flag_night_counter)
            cap.set(1 , set_pos)
            initial_flag = 1

    else:
        if flag_night_counter > 4:
            cv2.putText(frame,"NIGHT",(width - 200 ,50), font, 2,(167,133,0),2,cv2.LINE_AA)                   # NIGHT TIME
            mask1 = cv2.inRange(hsv, startRedLower, startRedUpper)
            mask2 = cv2.inRange(hsv, endRedLower, endRedUpper)
            maskRed = mask1 + mask2
            maskRed = cv2.erode(maskRed, None, iterations=2)
            maskRed = cv2.dilate(maskRed, None, iterations=2)
            # cv2.imshow('Red',imutils.resize(maskRed,width=250))

            (_, contours , hierarchy) = cv2.findContours(maskRed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            hull = []
            indHull = []
            for i in range(len(contours)):
                contourVar = contours[i]
                chull = cv2.convexHull(contourVar , False)
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
                    # cv2.circle(frame, (int(cX), int(cY)), int(radius),(167,133,0), 2)
                    indHull.append(i)
                    cv2.putText(frame,"Let me show you brake-lights radiations patches.",(170 ,80), font, 1.2,(0, 255, 255),2,cv2.LINE_AA)
                    cv2.putText(frame,"Apply brakes accordingly.",(330 ,120), font, 1.2,(0, 255, 255),2,cv2.LINE_AA)


            # print(indHull , len(indHull))

            # draw contours and hull points
            for i in indHull:
                # draw ith contour
                cv2.drawContours(frame, contours, i, (0, 255, 0), 1, 8, hierarchy)
                # draw ith convex hull object
                cv2.drawContours(frame, hull, i, (0, 255, 255), 2, 8)    
        else:                                                                                                   # DAY TIME
            cv2.putText(frame,"DAY",(width - 200 ,50), font, 2,(167,133,0),2,cv2.LINE_AA)


        fps.update()
    cv2.imshow("Frame", frame)
    # out1.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


fps.stop()
# out1.release()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
cv2.destroyAllWindows()





# a.mp4(25fps)   25*25    292*25    
# c.mp4(24fps) 133*24 + 6
# d.mp4(24fps) 195*24   419*24
# e.mp4(24fps) 47  283    338
# f.mp4(24fps) 36  223
# j.mp4(30fps)    false results
# k.mp4(30fps) blbwcv
# l.mp4(30fps)   54

# startRedLower = (0 , 150 , 50)
# startRedUpper = (15 , 255, 255)
# endRedLower = (165 , 150 , 50)
# endRedUpper = (180 , 255 , 255)