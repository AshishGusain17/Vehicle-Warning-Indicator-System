import numpy as np
import cv2
import sys
from sklearn import linear_model
from imutils.video import FPS




def preProcessImage(rgbImage):

    img_gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2. cvtColor(rgbImage, cv2.COLOR_BGR2HLS)
    ysize, xsize =  img_gray.shape[0], img_gray.shape[1]

    #Detecting yellow and white colors
    low_yellow = np.array([20, 100, 100])
    high_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, low_yellow, high_yellow)
    mask_white = cv2.inRange(img_gray, 200, 255)

    mask_yw = cv2.bitwise_or(mask_yellow, mask_white)
    mask_onimage = cv2.bitwise_and(img_gray, mask_yw)

    #Smoothing for removing noise
    gray_blur = cv2.GaussianBlur(mask_onimage, (5,5), 0)
    return gray_blur, xsize, ysize
 


def extractRegionOfInterest( preProcessedImage, mask_color):
    """ Extracts the Region of Interest based on Mask parameters
    @preProcessedImage: Image after preprocessing
    @mask_color: Mask color to be superimposed on Lanes
    """
    
    ysize, xsize = preProcessedImage.shape[0], preProcessedImage.shape[1]
    left_bottom = [0, ysize]
    right_bottom = [xsize-0, ysize]
    apex_left = [((xsize/2)-50), ((ysize/2)+50)]
    apex_right = [((xsize/2)+50), ((ysize/2)+50)]

    #Region of Interest Extraction
    mask_roi = np.zeros(preProcessedImage.shape, dtype=np.uint8) 
    roi_corners = np.array([[left_bottom, apex_left, apex_right, right_bottom]], dtype=np.int32)
    cv2.fillPoly(mask_roi, roi_corners, mask_color)
    image_roi = cv2.bitwise_and(preProcessedImage, mask_roi)

    #Thresholding before edge
    ret, img_postthresh = cv2.threshold(image_roi, 50, 255, cv2.THRESH_BINARY)
    return img_postthresh





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




cap = cv2.VideoCapture('../../videos/d.mp4')  
# cap.set(1,1700)  
fps = FPS().start()

#   Variables for deciding the color of Region of Interest
MASK_COLOR = 255

#   Edge Detection Variables
EDGE_LOW = 50
EDGE_HIGH = 200

#   Line Parameters
MINIMUM_LENGTH = 20
MAXIMUM_LINE_GAP = 10
LINE_DRAW_OPTION = 0

#   Execute while loop till video is open
while(cap.isOpened()):
    ret, frame = cap.read()

    #Escape when no frame is captured / End of Video
    if frame is None:
        break
    
    #   Preprocess Image    
    preProcessedImage, xsize, ysize = preProcessImage(frame)
    cv2.imshow('preProcessedImage',preProcessedImage)

    #   Extract the Region of Interest
    roiExtractedImage = extractRegionOfInterest(preProcessedImage, MASK_COLOR)
    cv2.imshow('roiExtractedImage',roiExtractedImage)


    lines = cv2.HoughLinesP(roiExtractedImage, 1, np.pi/180, 180, np.array([]), minLineLength = 5, maxLineGap = 35)
    draw_lines(roiExtractedImage,lines,frame)



    # cv2.imshow('Image',combinedImage)
    # cv2.imshow('smallGray',smallGray)

    # cv2.imshow('smallThreshold',smallThreshold)
    cv2.imshow("roiExtractedImage",roiExtractedImage)
    fps.update()
    key=cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break
    
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
# out1.release()
cv2.destroyAllWindows() 


