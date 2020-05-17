import cv2
from imutils.video import FPS
import copy

refPt = []
def click_and_crop(event, x, y, flags, param):
  global refPt
  # if the left mouse button was clicked, record the starting (x, y) coordinates
  if event == cv2.EVENT_LBUTTONDOWN:
    refPt = [(x, y)]

  # if the left mouse button was released, store the ending (x, y) coordinates
  elif event == cv2.EVENT_LBUTTONUP:
    refPt.append((x, y))
    cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)


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
    cv2.putText(image, "Drag and drop",  (60,30), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)
    cv2.putText(image, "Press 'r' to retry and 'd' when you are done cropping.",  (60,80), cv2.FONT_HERSHEY_PLAIN, 2, [0,255,255], 3)

    cv2.imshow("frame", image)
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
      image = clone.copy()
    # if the 'd' key is pressed, break from the loop
    elif key == ord("d"):
      if len(refPt)==2:
        cropped = 1
        # roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        # cv2.imshow("ROI", roi)
      cap.set(1,start_frame)
      fps = FPS().start()
    elif key == ord("q"):
      break
  else:
    print(refPt)
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
