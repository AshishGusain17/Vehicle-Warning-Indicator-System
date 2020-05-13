from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
from sklearn.metrics import pairwise
from PIL import ImageGrab
from imutils.video import FPS
import copy

# def draw_lines(img, lines,store):
# 	height , width = img.shape
# 	try:
# 		for line in lines:
# 			coords = line[0]
# 			cv2.line(store, (coords[0],coords[1]), (coords[2],coords[3]), [0,255,255], 3)         # yellow color vertical
# 	except:
# 	    pass
# 	    print("exception")
# 	cv2.imshow("store",store)


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


def draw_lines(img, lines,store):
  height , width = img.shape
  try:
    for line in lines:
      coords = line[0]
      flag = 0 
      if coords[2]==coords[0]:
        slope=99
        print("infinite slope")
        cv2.line(store, (coords[0],coords[1]), (coords[2],coords[3]), [0,255,255], 3)         # yellow color vertical
      else:
        slope=(coords[1]-coords[3])/(coords[2]-coords[0])
        if -0.3 < slope < 0.3:
          cv2.line(store, (coords[0],coords[1]), (coords[2],coords[3]), [255,0,0], 2)           # blue color horizontal
        else:
          if slope < 0:
            if (coords[0] + coords[2])/2 < width//2 < max([coords[0],coords[2]]):
              flag=1 
              slope=str(slope)[:5]
              cv2.putText(store, str(slope),  (coords[0],coords[1]), cv2.FONT_HERSHEY_PLAIN, 3, [122,32,12], 2)
              cv2.line(store, (coords[0],coords[1]), (coords[2],coords[3]), [0,0,0], 2)         # black color vertical
              cv2.putText(store, "get to your lane" ,  (40,40), cv2.FONT_HERSHEY_PLAIN, 3, [23,64,21], 3)

          else:
            if (coords[0] + coords[2])/2 < width//2 < min([coords[0],coords[2]]):
              flag=1 
              slope=str(slope)[:5]
              cv2.putText(store, str(slope),  (coords[0],coords[1]), cv2.FONT_HERSHEY_PLAIN, 3, [122,32,12], 2)
              cv2.line(store, (coords[0],coords[1]), (coords[2],coords[3]), [0,0,0], 2)         # black color vertical
              cv2.putText(store, "get to your lane" ,  (40,40), cv2.FONT_HERSHEY_PLAIN, 3, [23,64,21], 3)

          if flag != 1:
            slope=str(slope)[:5]
            cv2.putText(store, str(slope),  (coords[0],coords[1]), cv2.FONT_HERSHEY_PLAIN, 3, [122,32,12], 2)
            cv2.line(store, (coords[0],coords[1]), (coords[2],coords[3]), [0,255,255], 2)         # yellow color vertical


  except:
      pass
  cv2.imshow("store",store)





def process_img(image):
    # convert to gray
    store = copy.deepcopy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    image =  cv2.Canny(image, threshold1 = 200, threshold2=300)
    image = cv2.GaussianBlur(image,(3,3),0)

    # vertices = np.array(refPt, np.int32)
    vertices = np.array([[197, 537], [593, 339], [725, 335], [1271, 365], [1271, 592], [184, 550]], np.int32)
    image = roi(image, [vertices])

    lines = cv2.HoughLinesP(image, 1, np.pi/180, 180, np.array([]), minLineLength = 5, maxLineGap = 5)
    draw_lines(image,lines,store)
    return image
    

def selectRegions(image , cropped , str , key):
    global refPt
    while True:
      # display the image and wait for a keypress
      cv2.putText(image, str ,  (60,30), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)
      cv2.putText(image, "Press 'r' key to reset everything.",  (60,70), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)
      cv2.putText(image, "Press 'd' key if the region selection is done.",  (60,110), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)

      for pt in range(len(refPt)-1):
        pt1 , pt2 = refPt[pt] , refPt[pt+1]
        cv2.line(image, (pt1[0],pt1[1]), (pt2[0],pt2[1]), [0,255,255], 3)      

      cv2.imshow("ROI", image)
      if key == ord("r"):
        image = clone.copy()
        refPt = []
      elif key == ord("d"):
        if len(refPt) > 2:
          cropped = 1
          vertices = np.array(refPt, np.int32)
          image = roi(clone, [vertices])
          cv2.imshow("ROI", image)
          cap.set(1,start_frame)
          fps = FPS().start()
          print(refPt)
      elif key == ord('q'):
        break

refPt = []
cropped = 0
cap=cv2.VideoCapture('../../videos/d.mp4')
start_frame = 1500
cap.set(1,start_frame)
_ , image = cap.read()
clone = copy.deepcopy(image)
cv2.namedWindow("ROI")
cv2.setMouseCallback("ROI", click_and_crop)
ctt = 0
while True:
  key = cv2.waitKey(1) & 0xFF
  if cropped == 0:
    selectRegions(image , cropped , "Click points to select region of interset" ,key)
    print(1)
  elif cropped == 1:
    selectRegions(image , cropped , "Click points to select bird's eye view",key)
  else:
    ctt = ctt + 1
    print(ctt)
    _,frame=cap.read()
    if _ == False:
      break

    new_screen = process_img(frame)
    cv2.imshow("frame",frame)
    fps.update()
  if key == ord('q'):
    break
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
cap.release()
