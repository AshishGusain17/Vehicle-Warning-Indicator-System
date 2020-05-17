from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
from sklearn.metrics import pairwise
from PIL import ImageGrab
from imutils.video import FPS
import copy




def click_and_crop(event, x, y, flags, param):
  global refPt
  # if the left mouse button was clicked, record the starting (x, y) coordinates
  if event == cv2.EVENT_LBUTTONDOWN:
    refPt.append([x, y])


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, [255,255,255])
    # cv2.imshow("mask",mask)
    masked = cv2.bitwise_and(img, mask)
    return masked



    

def selectRegions(image  , text , flag):
    global refPt , cropped
    clone = copy.deepcopy(image)
    while True:
      key = cv2.waitKey(1) & 0xFF
      # display the image and wait for a keypress
      cv2.putText(image, text ,  (60,30), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)
      cv2.putText(image, "Press 'r' key to reset everything.",  (60,70), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)
      cv2.putText(image, "Press 'd' key if the region selection is done.",  (60,110), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)

      for pt in range(len(refPt)-1):
        pt1 , pt2 = refPt[pt] , refPt[pt+1]
        cv2.line(image, (pt1[0],pt1[1]), (pt2[0],pt2[1]), [0,255,255], 3)      

      cv2.imshow("ROI", image)
      if key == ord("r"):
        image = copy.deepcopy(clone)
        refPt = []
      elif key == ord("d"):
        if len(refPt) > 2:
          cropped = flag
          vertices = np.array(refPt, np.int32)
          image = roi(clone, [vertices])
          cv2.imshow("ROI", image)
          cap.set(1,start_frame)
          return 
      elif key == ord('q'):
        return 1


refPt = []
cropped = 0
cap=cv2.VideoCapture('../../videos/d.mp4')
start_frame = 1500
cap.set(1,start_frame)
_ , image = cap.read()
cv2.namedWindow("ROI")
cv2.setMouseCallback("ROI", click_and_crop)
ctt = 0
Quit = 0
while True:
  key = cv2.waitKey(1) & 0xFF
  if cropped == 0:
    Quit = selectRegions(copy.deepcopy(image)  , "Click points to select region of interset" , 1)
    roiPointer = refPt
    refPt = []
    print(roiPointer)
  elif cropped == 1:
    Quit = selectRegions(copy.deepcopy(image)  , "Click points to select bird's eye view" , 2)
    lanePointer = refPt
    print(lanePointer)
    fps = FPS().start()
  else:
    _,frame = cap.read()
    if _ == False:
      break
    ctt = ctt + 1
    print(ctt)

    cv2.imshow("frame",frame)
    fps.update()
    if key == ord('q'):
      fps.stop()
      print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
      print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
      break
  if Quit:
    break

cv2.destroyAllWindows()
cap.release()
