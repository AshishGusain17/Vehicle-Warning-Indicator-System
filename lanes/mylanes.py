from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
from sklearn.metrics import pairwise
from PIL import ImageGrab

height , width , channels=0,0,0

def draw_lines(img, lines,store):
	try:
		for line in lines:
			coords = line[0]
			flag = 0 
			if coords[2]==coords[0]:
				slope=99
			else:
				slope=(coords[1]-coords[3])/(coords[2]-coords[0])
				if -0.3 < slope < 0.3:
					cv2.line(store, (coords[0],coords[1]), (coords[2],coords[3]), [255,0,0], 3)           # blue color horizontal
				else:
					# if         width//2 abs(coords[0] - coords[2])
					if slope < 0:
						if  (coords[0] + coords[2])/2 < width//2 < max([coords[0],coords[2]]):
							flag=1
					else:
						if (coords[0] + coords[2])/2 < width//2 < min([coords[0],coords[2]]):
							flag=1

					slope=str(slope)[:5]
					cv2.putText(store, str(slope),  (coords[0],coords[1]), cv2.FONT_HERSHEY_PLAIN, 3, [122,32,12], 2)
					cv2.line(store, (coords[0],coords[1]), (coords[2],coords[3]), [0,255,255], 3)         # yellow color vertical

					if flag == 1:
						cv2.putText(store, "get to your lane" ,  (40,40), cv2.FONT_HERSHEY_PLAIN, 3, [23,64,21], 3)


			# cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [0,255,255], 5)
			# cv2.line(store, (coords[0],coords[1]), (coords[2],coords[3]), [0,255,255], 3)


	except:
	    pass
	cv2.imshow("store",store)
	# cv2.imshow("later",img)



def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    cv2.imshow("poly",mask)
    masked = cv2.bitwise_and(img, mask)

    return masked

def process_img(image):
    # convert to gray
    store=image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    image =  cv2.Canny(image, threshold1 = 200, threshold2=300)
    image = cv2.GaussianBlur(image,(3,3),0)

    # vertices = np.array([[50,500],[50,300], [450,200], [800,200], [1230,300], [1230,500]], np.int32)
    # vertices = np.array([[50,500],[50,350], [450,250], [800,250], [1230,350], [1230,500]], np.int32)
    # vertices = np.array([[80,500],[80,370], [450,270], [800,270], [1200,370], [1200,500]], np.int32)
    # vertices = np.array([[160,630],[160,370], [450,280], [800,280], [1120,370], [1120,630]], np.int32)
    vertices = np.array([[160,630],[160,420], [450,280], [800,280], [1120,420], [1120,630]], np.int32)


    image = roi(image, [vertices])

    lines = cv2.HoughLinesP(image, 1, np.pi/180, 180, np.array([]), minLineLength = 5, maxLineGap = 35)
    draw_lines(image,lines,store)

    return image
    




# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('../../videos/c.mp4')
cap.set(1,1700)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out1 = cv2.VideoWriter('MI_V-s_CSK.avi', fourcc, 10.0, (int(cap.get(3)),int(cap.get(4))))



def screen_record(): 
	last_time = time.time()
	while(True):
		_ , screen =  cap.read()
		global height , width , channels
		height, width, channels = screen.shape
		cv2.imshow('original', screen )

		new_screen = process_img(screen)
		# print('loop took {} seconds'.format(time.time()-last_time))
		# last_time = time.time()
		cv2.imshow('later', new_screen )
		if cv2.waitKey(25) & 0xFF == ord('q'):
		    cv2.destroyAllWindows()
		    cap.release()
		    break


screen_record()
