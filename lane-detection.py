from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
from sklearn.metrics import pairwise
from PIL import ImageGrab
from imutils.video import FPS
import copy

font = cv2.FONT_HERSHEY_SIMPLEX


# def all_lines(img, lines, store):
# 	height, width = img.shape
# 	try:
# 		for line in lines:
# 			coords = line[0]
# 			cv2.line(store, (coords[0], coords[1]), (coords[2], coords[3]), [0, 255, 255], 3)         # yellow color vertical
# 	except:
# 		pass
# 		print("exception")
# 	cv2.imshow("store", store)






def click_and_crop(event, x, y, flags, param):
    global refPt
    # if the left mouse button was clicked, record the starting (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append([x, y])



def roi(img, vertices):
	mask = np.zeros_like(img)
	cv2.fillPoly(mask, vertices, [255, 255, 255])
	# cv2.imshow("mask",mask)
	masked = cv2.bitwise_and(img, mask)
	return masked



def draw_lines(lanePointer , dashPointer , lane_image , image_np , flagLanes):
	height , width , channels= image_np.shape
	gray_image = cv2.cvtColor(lane_image , cv2.COLOR_BGR2GRAY)
	canny_image =  cv2.Canny(gray_image, threshold1 = 100 , threshold2 = 100)
	cv2.imshow("entire canny",canny_image)
	canny_image = cv2.GaussianBlur(canny_image,(3,3),0)

	mask = np.zeros_like(canny_image)
	vertices = np.array(lanePointer, np.int32)
	cv2.fillPoly(mask, [vertices], [255,255,255])

	cv2.imshow("mask",mask)
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
					# cv2.line(lane_image, (x1 , y1), (x2 , y2), [255,0,0], 2)                        # blue color horizontal
					justcomment = 0
				elif slope < 0:
					if width//2 > max(x1 , x2):
						slope=str(slope)[:5]
						# cv2.putText(lane_image, str(slope),  (x1 , y1), font, 3, [122,32,12], 2)
						cv2.line(lane_image, (x1 , y1), (x2 , y2), [0,0,0], 2)                      # black color vertical
						flagCounter = 1
					else:
						slope=str(slope)[:5]
						# cv2.putText(lane_image, str(slope),  (x1 , y1), font, 3, [122,32,12], 2)
						cv2.line(lane_image, (x1 , y1), (x2 , y2), [0,255,255], 2)                   # yellow color vertical

				elif slope > 0:
					if width//2 < min(x1 , x2):
						slope=str(slope)[:5]
						# cv2.putText(lane_image, str(slope),  (x1 , y1), font, 3, [122,32,12], 2)
						cv2.line(lane_image, (x1 , y1), (x2 , y2), [0,0,0], 2)                       # black color vertical
						flagCounter = 1
					else:
						slope=str(slope)[:5]
						# cv2.putText(lane_image, str(slope),  (x1 , y1), font, 3, [122,32,12], 2)
						cv2.line(lane_image, (x1 , y1), (x2 , y2), [0,255,255], 2)                   # yellow color vertical
		if flagCounter == 1:
			flagLanes.append(1)
		else:
			flagLanes.append(0)
		if sum(flagLanes) > 12:
			cv2.putText(lane_image, "Get back to your lane" ,  (370,80), font , 1.2, (0,255,0), 2,cv2.LINE_AA)


	except:
		pass
	cv2.imshow("lane_image",lane_image)
	# out1.write(lane_image)



cap=cv2.VideoCapture('../videos/r.mp4')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out1 = cv2.VideoWriter('lanes.avi', fourcc, 25, (1280,720))
start_frame = 0*24
flagLanes = [0] * 20






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
            elif flag == 2:
                return 0
        elif key == ord('q'):
            return 1








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
	if len(lanePointer) <= 2:
		lanePointer = [[114, 690], [502, 384], [819, 391], [1201, 695]]
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
		lane_image = copy.deepcopy(frame)
		draw_lines(lanePointer , dashPointer , lane_image , frame , flagLanes)
		cv2.imshow("frame", frame)

		key = cv2.waitKey(1) & 0xFF
		fps.update()
		if key == ord('q'):
			break

	fps.stop()
	print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))





refPt = []                  # to store refernece pointers
flag_night_counter = 0      # counter to count night frames

cap.set(1,start_frame)
_ , image = cap.read()
image=imutils.resize(image, width=1280)
cv2.namedWindow("ROI")
cv2.setMouseCallback("ROI", click_and_crop)

cap.set(1 , start_frame)
day()



cv2.destroyAllWindows()
cap.release()
# out1.release()











# lanes   r
# a   451(lanes showing good)
# b   115(warning shows good )
# d   0
# d   81






