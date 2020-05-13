import cv2
from imutils.video import FPS
import copy
import numpy as np

refPt = []
def click_and_crop(event, x, y, flags, param):
  global refPt
  # if the left mouse button was clicked, record the starting (x, y) coordinates
  if event == cv2.EVENT_LBUTTONDOWN:
    refPt.append([x, y])

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, [255,255,255])
    cv2.imshow("mask",mask)
    masked = cv2.bitwise_and(img, mask)
    return masked

cropped = 0
cap=cv2.VideoCapture('../../videos/highway.mp4')
start_frame = 990
cap.set(1,start_frame)
_ , image = cap.read()
clone = copy.deepcopy(image)
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", click_and_crop)
while True:
  key = cv2.waitKey(1) & 0xFF
  if cropped == 0:
    # display the image and wait for a keypress
    cv2.putText(image, "Click points to select region of interest.",  (60,30), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)
    cv2.putText(image, "Press 'r' key to reset everything.",  (60,70), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)
    cv2.putText(image, "Press 'd' key if the region selection is done.",  (60,110), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)

    for pt in range(len(refPt)-1):
      pt1 , pt2 = refPt[pt] , refPt[pt+1]
      cv2.line(image, (pt1[0],pt1[1]), (pt2[0],pt2[1]), [0,255,255], 3)      

    cv2.imshow("frame", image)
    # if the 'r' key is pressed, reset the cropping region
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
    elif key == ord("q"):
      break

  else:
    _,frame=cap.read()
    if _ == False:
      break
    cv2.imshow("frame",frame)
    fps.update()
    if key == ord('q'):
        break
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
cap.release()
