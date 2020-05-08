import numpy as np
import cv2
import imutils
import time
from imutils.video import FPS
from sklearn.metrics import pairwise



font = cv2.FONT_HERSHEY_SIMPLEX

startRedLower = (0 , 200 , 80)
startRedUpper = (7 , 255, 255)
endRedLower = (173 , 200 , 80)
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
    if ratio < 0.55:
        flag_night_counter = flag_night_counter + 1
        return flag_night_counter
    else:
        flag_night_counter = flag_night_counter - 1 
        return flag_night_counter


# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('../videos/f.mp4')
set_pos=72*25
cap.set(1,set_pos)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('MI_V-s_CSK.avi', fourcc, 10.0, (int(cap.get(3)),int(cap.get(4))))
fps = FPS().start()

ct=0
flag_night_counter = 0
initial_flag = 0
while True:
    (grabbed, frame) = cap.read()
    height,width,channel = frame.shape
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
            cv2.imshow('Red',imutils.resize(maskRed,width=250))

            (_, contours , hierarchy) = cv2.findContours(maskRed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            hull = []
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
                center=(cX,cY)

                if radius > 10:
                    cv2.circle(frame, (int(cX), int(cY)), int(radius),(0, 255, 255), 2)
                    hull.append(chull)

            print("length = ",len(hull))

            # draw contours and hull points
            for i in range(len(hull)):
                color_contours = (0, 255, 0) # green - color for contours
                color = (255, 0, 0) # blue - color for convex hull
                # draw ith contour
                cv2.drawContours(frame, contours, i, color_contours, 1, 8, hierarchy)
                # draw ith convex hull object
                cv2.drawContours(frame, hull, i, color, 1, 8)    
        else:                                                                                                   # DAY TIME
            cv2.putText(frame,"DAY",(width - 200 ,50), font, 2,(167,133,0),2,cv2.LINE_AA)



        fps.update()
    cv2.imshow("Frame", frame)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break


fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
cv2.destroyAllWindows()





# a.mp4   60*25

# startRedLower = (0 , 150 , 50)
# startRedUpper = (15 , 255, 255)
# endRedLower = (165 , 150 , 50)
# endRedUpper = (180 , 255 , 255)