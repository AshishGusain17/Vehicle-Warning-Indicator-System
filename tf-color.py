from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from sklearn.metrics import pairwise


size=64
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (35, 86, 6)
greenUpper = (100, 255, 255)

yellowLower = (14, 76, 6)
yellowUpper = (34, 255, 255)

startRedLower = (0 , 50 , 6)
startRedUpper = (15 , 255, 255)
endRedLower = (150 , 50 , 6)
endRedUpper = (180 , 255 , 255)





# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('../videos/f.mp4')
cap.set(1,1100)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('MI_V-s_CSK.avi', fourcc, 10.0, (int(cap.get(3)),int(cap.get(4))))


time.sleep(2.0)

ct=0
while True:
    (grabbed, frame) = cap.read()
    print('frame',frame.shape)

    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    print('frame_resize',frame.shape)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    print('blurred',blurred.shape)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    print('hsv',hsv.shape)
    print()

    for ii in range(10):
        er=0

    # create a mask containing 0 and 255 values only



    mask1 = cv2.inRange(hsv, startRedLower, startRedUpper)
    mask2 = cv2.inRange(hsv, endRedLower, endRedUpper)
    maskRed = mask1 + mask2
    # print(mask)
    maskRed = cv2.erode(maskRed, None, iterations=2)
    maskRed = cv2.dilate(maskRed, None, iterations=2)
    cv2.imshow('Red',imutils.resize(maskRed,width=250))
    (_, cnts, _) = cv2.findContours(maskRed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)



    maskYellow = cv2.inRange(hsv, yellowLower, yellowUpper)
    maskYellow = cv2.erode(maskYellow, None, iterations=2)
    maskYellow = cv2.dilate(maskYellow, None, iterations=2)
    cv2.imshow('Yellow',imutils.resize(maskYellow,width=250))
    (_, cnts, _) = cv2.findContours(maskYellow.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)



    maskGreen = cv2.inRange(hsv, greenLower, greenUpper)
    maskGreen = cv2.erode(maskGreen, None, iterations=2)
    maskGreen = cv2.dilate(maskGreen, None, iterations=2)
    cv2.imshow('Green',imutils.resize(maskGreen,width=250))
    (_, cnts, _) = cv2.findContours(maskGreen.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)




    cv2.imshow("Frame", frame)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()