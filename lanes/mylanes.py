from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time
from sklearn.metrics import pairwise
from PIL import ImageGrab

def draw_lines(img, lines,store):
	try:
		for line in lines:
			coords = line[0]
			cv2.line(store, (coords[0],coords[1]), (coords[2],coords[3]), [0,255,255], 5)

	except:
	    pass
	cv2.imshow("store",store)



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
    # image = cv2.GaussianBlur(image,(2,2),0)

    # vertices = np.array([[50,500],[50,300], [450,200], [800,200], [1230,300], [1230,500]], np.int32)
    # vertices = np.array([[50,500],[50,350], [450,250], [800,250], [1230,350], [1230,500]], np.int32)
    # vertices = np.array([[80,500],[80,370], [450,270], [800,270], [1200,370], [1200,500]], np.int32)
    # vertices = np.array([[160,630],[160,370], [450,280], [800,280], [1120,370], [1120,630]], np.int32)
    vertices = np.array([[160,630],[160,420], [450,280], [800,280], [1120,420], [1120,630]], np.int32)


    image = roi(image, [vertices])

    lines = cv2.HoughLinesP(image, 1, np.pi/180, 180,  minLineLength = 10, maxLineGap = 15)
    draw_lines(image,lines,store)

    return image
    




# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('../../videos/e.mp4')
cap.set(1,200)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out1 = cv2.VideoWriter('MI_V-s_CSK.avi', fourcc, 10.0, (int(cap.get(3)),int(cap.get(4))))



def screen_record(): 
    last_time = time.time()
    while(True):
        _ , screen =  cap.read()
        cv2.imshow('original', screen )

        new_screen = process_img(screen)
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('later', new_screen )
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break

screen_record()
