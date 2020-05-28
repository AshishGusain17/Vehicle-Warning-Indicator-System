import numpy as np
import cv2
import imutils
import time
from sklearn.metrics import pairwise
from imutils.video import FPS
import copy
import os
import sys
import tensorflow as tf
import pathlib
from collections import defaultdict

from utils import tracking_utils 
from utils import signalDetection_utils 
from utils import estimate_collide_utils
from utils import estimate_stepping_utils
from utils import lane_detection_utils
from utils import break_light_utils
from utils import functions

from utils import ops as utils_ops
from utils import label_map_util



utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
PATH_TO_LABELS = '../bigdata/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)




model_name = 'ssd_inception_v2_coco_2018_01_28'
model_dir =  "../bigdata/models/" + model_name + "/saved_model"
detection_model = tf.saved_model.load(str(model_dir))
detection_model = detection_model.signatures['serving_default']



# print(category_index)
colors = np.random.uniform(0, 255, size=(len(category_index), 3))
font = cv2.FONT_HERSHEY_SIMPLEX
blackLower = (0 , 0 , 0)
blackUpper = (180 , 255 , 35)

print(detection_model.inputs)
print(detection_model.output_dtypes)
print(detection_model.output_shapes)








def click_and_crop(event, x, y, flags, param):
    global refPt
    # if the left mouse button was clicked, record the starting (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append([x, y])





def confirm_day_or_night(frame , flag_night_counter):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blackLower , blackUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask , None, iterations=2)
    # cv2.imshow('black',imutils.resize(mask,width=250))
    # cv2.imshow('frame',frame)
    pixel_ct = 0
    pixel_len = 0
    for i in mask:
      pixel_ct = pixel_ct + np.sum(i==0)
      pixel_len = pixel_len + len(i)
    ratio = pixel_ct / pixel_len
    print("ratio = ",ratio)
    if ratio < 0.5:
        flag_night_counter = flag_night_counter + 1
        return flag_night_counter
    else:
        flag_night_counter = flag_night_counter - 1 
        return flag_night_counter




# flagPerson and areaPerson are pedestrian crash
flagPerson = 0
areaPerson = 0
areaDetails = []

# crash_count_frames are vehicle warning variables
crash_count_frames = 0

# signalCounter and flagSignal are traffic signal variables
signalCounter = -99999
flagSignal = [0] * 20

# number and prev_frame are tracking variables
number = 0
prev_frame = []
def show_inference(dashPointer , lanePointer , frame):
    global flagPerson , areaPerson , areaDetails
    global crash_count_frames
    global signalCounter , flagSignal
    global prev_frame , number

    image_np = np.array(frame)
    lane_image = copy.deepcopy(image_np)
    height , width , channel = image_np.shape


    output_dict = functions.get_dict(dashPointer , detection_model , image_np)

    confidencesCars , boxesCars , confidencesLights , boxesLights , confidencesPersons , boxesPersons = functions.findBoxes(image_np , output_dict)

    indexesLights = cv2.dnn.NMSBoxes(boxesLights, confidencesLights, 0.5, 0.4)
    indexesCars = cv2.dnn.NMSBoxes(boxesCars, confidencesCars, 0.5, 0.4)
    indexesPersons = cv2.dnn.NMSBoxes(boxesPersons, confidencesPersons, 0.5, 0.4)

    image_np , signalCounter , flagSignal = signalDetection_utils.signalDetection(indexesLights , boxesLights , image_np , signalCounter , flagSignal)
    image_np , prev_frame , number = tracking_utils.tracking(indexesCars , boxesCars , image_np , prev_frame , number)
    image_np , crash_count_frames = estimate_collide_utils.estimate_collide(indexesCars , boxesCars , image_np , crash_count_frames)
    image_np , flagPerson , areaPerson , areaDetails = estimate_stepping_utils.estimate_stepping(indexesPersons , boxesPersons , image_np, flagPerson , areaPerson , areaDetails)

    cv2.putText(image_np,"DAY",(50 ,90), font, 2, (167,133,0) , 2 , cv2.LINE_AA)

    image_np = lane_detection_utils.draw_lines(lanePointer , lane_image , image_np)
    # lane_detection_utils.all_lines(lanePointer , lane_image , image_np)

    cv2.imshow("finally", image_np)





def selectRegions(image  , text , flag):
    global refPt
    clone = copy.deepcopy(image)
    while True:
        key = cv2.waitKey(1) & 0xFF
        # display the image and wait for a keypress
        if flag==1:
            cv2.putText(image, text ,  (240,30), font , 1.2, [0,255,255], 2,cv2.LINE_AA)
            cv2.putText(image, "Press 'r' key to reset everything.",  (290,70), font , 1.2, [0,255,255], 2,cv2.LINE_AA)
            cv2.putText(image, "Press 'd' key if the region selection is done.",  (180,110), font , 1.2, [0,255,255], 2,cv2.LINE_AA)
        else:
            cv2.putText(image, text ,  (240,30), font , 1.2, [0,255,0], 2,cv2.LINE_AA)
            cv2.putText(image, "Press 'r' key to reset everything.",  (290,70), font , 1.2, [0,255,0], 2,cv2.LINE_AA)
            cv2.putText(image, "Press 'd' key if the region selection is done.",  (180,110), font , 1.2, [0,255,0], 2,cv2.LINE_AA)

        for pt in range(len(refPt)-1):
            pt1 , pt2 = refPt[pt] , refPt[pt+1]
            cv2.line(image, (pt1[0],pt1[1]), (pt2[0],pt2[1]), [0,255,255], 3)      

        cv2.imshow("ROI", image)
        if key == ord("r"):
            image = copy.deepcopy(clone)
            refPt = []
        elif key == ord("d"):
            if flag == 1:
                return 0
            elif flag == 2 and len(refPt) > 2:
                return 0
        elif key == ord('q'):
            return 1







def night():
    global refPt
    _ , image = cap.read()
    image=imutils.resize(image, width=1280)

    ctt = 0

    Quit = selectRegions(image  , "Click points to select your vehicle dash." , 1)
    dashPointer = refPt
    if len(dashPointer) <= 2:
        dashPointer = [[0,0], [0,0], [0,0]]
    refPt = []
    print("For dash: ",dashPointer)
    if Quit == 1:
        return
    cv2.destroyWindow("ROI")

    fps = FPS().start()
    while True:
        _,frame = cap.read()
        frame=imutils.resize(frame, width=1280)

        if _ == False:
            break

        # print(ctt ,fps._numFrames)
        # ctt = ctt + 1
        break_light_utils.break_light(dashPointer , frame)

        # cv2.imshow("original",frame)
        key = cv2.waitKey(1) & 0xFF
        fps.update()
        if key == ord('q'):
            break
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))




def day():
    global refPt
    _ , image = cap.read()
    image=imutils.resize(image, width=1280)

    ctt = 0

    Quit = selectRegions(copy.deepcopy(image)  , "Click points to select your vehicle dash." , 1)
    dashPointer = refPt
    if len(dashPointer) <= 2:
        dashPointer = [[0,0], [0,0], [0,0]]
    refPt = []
    print("For dash: ",dashPointer)
    if Quit == 1:
        return

    Quit = selectRegions(copy.deepcopy(image)  , "Click points to select bird's eye view." , 2)
    lanePointer = refPt
    print("For lanes: ",lanePointer)
    if Quit == 1:
        return

    cv2.destroyWindow("ROI")

    fps = FPS().start()
    while True:
        _,frame = cap.read()
        frame = imutils.resize(frame, width=1280)
        if _ == False:
            break
        # print(ctt ,fps._numFrames)
        # ctt = ctt + 1

        show_inference(dashPointer , lanePointer , frame)

        # cv2.imshow("original",frame)
        key = cv2.waitKey(1) & 0xFF
        fps.update()
        if key == ord('q'):
            break

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))





refPt = []                  # to store refernece pointers
flag_night_counter = 0      # counter to count night frames

cap=cv2.VideoCapture('../videos/c.mp4')
start_frame = 113*24
cap.set(1,start_frame)
_ , image = cap.read()
image=imutils.resize(image, width=1280)
cv2.namedWindow("ROI")
cv2.setMouseCallback("ROI", click_and_crop)

for z in range(10):
    (grabbed, frame) = cap.read()
    frame=imutils.resize(frame, width=1280)
    height,width,channel = frame.shape

    flag_night_counter = confirm_day_or_night(frame , flag_night_counter)
print("flag_night_counter = ",flag_night_counter)
cap.set(1 , start_frame)

if flag_night_counter > 0:
    night()
else:
    day()



cv2.destroyAllWindows()
cap.release()
