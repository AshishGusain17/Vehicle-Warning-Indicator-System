

import numpy as np
import cv2
import sys
# import rospy
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
from sklearn import linear_model
from imutils.video import FPS


def extract_lane(road_lines):
    """ Extracts the left and right
    road lanes
    @road_lines: Array of x,y co-ordinates of line
    """

    left_lane = []
    right_lane = []
    left_slope = []
    right_slope = []

    if road_lines is not None:
        for x in range(0, len(road_lines)):
            for x1,y1,x2,y2 in road_lines[x]:
                slope = compute_slope(x1,y1,x2,y2)
                if slope=="nan":
                    pass
                elif (slope < 0):
                    left_lane.append(road_lines[x])
                    left_slope.append(slope)
                else:
                    if (slope > 0):
                        right_lane.append(road_lines[x])
                        right_slope.append(slope)
                
    return left_lane, right_lane , left_slope, right_slope
 

def compute_slope(x1,y1,x2,y2):
    """ Computes the slope from 
    x and y co-ordinates of two points
    @x1: x-intercept of line1
    @y1: y-intercept of line1
    @x2: x-intercept of line2
    @y2: y-intercept of line2"""

    if x2!=x1:
        return ((y2-y1)/(x2-x1))
    else:
        return "nan"



def print_lanes(left_lane, right_lane, left_slope, right_slope):
    """ Prints lane slope and intercept values for debug purposes.
    @left_lane: Left Lane Intercept Values
    @right_lane: Right Lane Intercept Values
    @left_slope: Left Lane Slope
    @right_slope: Right Lane Slope"""

    #print("Left lane")
    for x in range(0, len(left_lane)):
        print(left_lane[x], left_slope[x])
    #print("Right lane")
    for x in range(0, len(right_lane)):
        print(right_lane[x], right_slope[x])



def split_append(left_lane, right_lane):
    """ Converts the lane array from two points of a line
    to a single array with x and y co-ordinates for 
    left and right side each.
    @left_lane: Left Lane indivisual line Intercept Values
    @right_lane: Right Lane indivisual line Intercept Values
    """

    left_lane_sa = []
    right_lane_sa = []
    
    for x in range(0, len(left_lane)):
        for x1,y1,x2,y2 in left_lane[x]:
            left_lane_sa.append([x1, y1])
            left_lane_sa.append([x2, y2])

    for y in range(0, len(right_lane)):
        for x1,y1,x2,y2 in right_lane[y]:
            right_lane_sa.append([x1,y1])
            right_lane_sa.append([x2,y2])
            
    left_lane_sa = np.array(left_lane_sa)
    right_lane_sa = np.array(right_lane_sa)
    left_lane_sa,right_lane_sa = sort(left_lane_sa,right_lane_sa)
    return left_lane_sa,right_lane_sa



def print_lanes_sa(left_lane_sa, right_lane_sa):
    """ Prints the lanes after the frame is split and merged
    @left_lane_sa: Left Lane values after split and Append method
    @right_lane_sa: Right Lane values after split and Append method
    """

    print("Left lane")
    for x in range(0, len(left_lane_sa)):
        print(left_lane_sa[x])
    print("Right lane")
    for x in range(0, len(right_lane_sa)):
        print(right_lane_sa[x])          



def sort(left_lane_sa,right_lane_sa):
    """ Sorts lane values for left and right lanes 
    @left_lane_sa: Left Lane values after split and Append method
    @right_lane_sa: Right Lane values after split and Append method
    """

    #print(len(right_lane_sa));
    if (len(left_lane_sa) != 0):
        left_lane_sa = left_lane_sa[np.argsort(left_lane_sa[:, 0])]
    else:
        np.append(left_lane_sa, [0,0])
    
    if (len(right_lane_sa) != 0):
        right_lane_sa = right_lane_sa[np.argsort(right_lane_sa[:, 0])]
    else:
        right_lane_sa = [[0,0],[0,0]]
    return left_lane_sa, right_lane_sa



def draw_lanes(left_lane_sa, right_lane_sa, frame):
    """ Draws simple Lines on Input RGB Image
    @left_lane_sa: Left Lane values after split and Append method
    @right_lane_sa: Right Lane values after split and Append method
    @frame: RGB Input Image
    """

    #print_lanes_sa(left_lane_sa, right_lane_sa)
    if(len(left_lane_sa)!=0 & len(right_lane_sa)!=0):

        (vx_left,vy_left,x0_left,y0_left) = cv2.fitLine(left_lane_sa,cv2.DIST_L2,0,0.01,0.01)
        (vx_right,vy_right,x0_right,y0_right) = cv2.fitLine(right_lane_sa,cv2.DIST_L2,0,0.01,0.01)
        left_len = len(left_lane_sa)
        right_len = len(right_lane_sa)
        slope_left = vy_left / vx_left
        slope_right = vy_right / vx_right
        intercept_left = y0_left - (slope_left * x0_left)
        intercept_right = y0_right - (slope_right * x0_right)

        ysize = frame.shape[0]
        xsize = frame.shape[1]
        y_limit_low = int(0.95*ysize)
        y_limit_high = int(0.65*ysize)

        #Coordinates for point 1(Bottom Left)
        y_1 = ysize
        x_1 = int((y_1-intercept_left)/slope_left)

        #Coordinates for point 2(Bottom Left)
        y_2 = y_limit_high
        x_2 = int((y_2-intercept_left)/slope_left)

        #Coordinates for point 3(Bottom Left)
        y_3 = y_limit_high
        x_3 = int((y_3-intercept_right)/slope_right)
        
        #Coordinates for point 4(Bottom Right)
        y_4 = ysize
        x_4 = int((y_4-intercept_right)/slope_right)

        #Draw lines
        cv2.line(frame,(x_1,y_1),(x_2,y_2),(0,255,255),3)
        cv2.line(frame,(x_3,y_3),(x_4,y_4),(0,255,255),3)
        pts = np.array([[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]])
        mask_color = (255,255,0)
        frame_copy = frame.copy()
        cv2.fillPoly(frame_copy, np.int32([pts]), mask_color)
        opacity = 0.4
        cv2.addWeighted(frame_copy,opacity,frame,1-opacity,0,frame)
        return frame



def ransac_drawlane(left_lane_sa, right_lane_sa,frame):
    """ Draws Lines using RANSAC on Input RGB Image
    @left_lane_sa: Left Lane values after split and Append method
    @right_lane_sa: Right Lane values after split and Append method
    @frame: RGB Input Image
    """

    left_lane_x = []
    left_lane_y = []
    right_lane_x = []
    right_lane_y = []

    for x1,y1 in left_lane_sa:
        left_lane_x.append([x1])
        left_lane_y.append([y1])

    for x1,y1 in right_lane_sa:
        right_lane_x.append([x1])
        right_lane_y.append([y1])

    left_ransac_x = np.array(left_lane_x)
    left_ransac_y = np.array(left_lane_y)

    right_ransac_x = np.array(right_lane_x)
    right_ransac_y = np.array(right_lane_y)

        
    left_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    #print(left_ransac_x,left_ransac_y,len(left_ransac_x),len(left_ransac_y), left_ransac_x.shape )
    left_ransac.fit(left_ransac_x, left_ransac_y)
    slope_left = left_ransac.estimator_.coef_
    intercept_left = left_ransac.estimator_.intercept_

    right_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    right_ransac.fit(right_ransac_x, right_ransac_y)
    slope_right = right_ransac.estimator_.coef_
    intercept_right = right_ransac.estimator_.intercept_

    ysize = frame.shape[0]
    xsize = frame.shape[1]
    y_limit_low = int(0.95*ysize)
    y_limit_high = int(0.65*ysize)

    #Coordinates for point 1(Bottom Left)
    y_1 = ysize
    x_1 = int((y_1-intercept_left)/slope_left)

    #Coordinates for point 2(Bottom Left)
    y_2 = y_limit_high
    x_2 = int((y_2-intercept_left)/slope_left)

    #Coordinates for point 3(Bottom Left)
    y_3 = y_limit_high
    x_3 = int((y_3-intercept_right)/slope_right)
    
    #Coordinates for point 4(Bottom Right)
    y_4 = ysize
    x_4 = int((y_4-intercept_right)/slope_right)

    cv2.line(frame,(x_1,y_1),(x_2,y_2),(0,255,255),3)
    cv2.line(frame,(x_3,y_3),(x_4,y_4),(0,255,255),3)
    pts = np.array([[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]])
    mask_color = (255,255,0)
    frame_copy = frame.copy()
    cv2.fillPoly(frame_copy, np.int32([pts]), mask_color)
    opacity = 0.4
    cv2.addWeighted(frame_copy,opacity,frame,1-opacity,0,frame)
    return frame
    


def preProcessImage(rgbImage):
    """ Preprocess the input RGB image
    @rgbImage: Input RGB Image
    """

    # Color space conversion
    img_gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2. cvtColor(rgbImage, cv2.COLOR_BGR2HLS)
    ysize, xsize = getShape(img_gray)

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
    
    ysize, xsize = getShape(preProcessedImage)
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



def drawLines(roiExtractedImage, minLength, maxGap, lineDrawOption):
    """ Extract HOugh Lanes from ROI Image
    @roiExtractedImage: ROI Extracted Image
    @minLength: Minimum Length HOugh Parameter
    @maxGap: Max Gap HOugh Parameter
    @lineDrawOption: 1- RANSAC 0- Normal
    """

    road_lines = cv2.HoughLinesP(roiExtractedImage, 1, np.pi/180, 20, minLength, maxGap)
    left_lane, right_lane, left_slope, right_slope = extract_lane(road_lines)
    left_lane_sa, right_lane_sa = split_append(left_lane, right_lane)
    if lineDrawOption == 0:
        outputDetectFrame = draw_lanes(left_lane_sa, right_lane_sa,frame)
    else:
        outputDetectFrame = ransac_drawlane(left_lane_sa, right_lane_sa,frame)    
    return outputDetectFrame


def getShape(image):
    """ Get x and y size
    @image: Input Image"""

    return image.shape[0], image.shape[1]

def getGrayImage(rgbImage):
    """ COnvert RGB to Gray
    @image: Input RGB Image"""

    return cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)             





cap = cv2.VideoCapture('../../videos/a.mp4')    
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

    #   Use canny edge detection
    edgeImage = cv2.Canny(roiExtractedImage, EDGE_LOW, EDGE_HIGH)
    cv2.imshow('edgeImage',edgeImage)

    #   Hough Line Draw
    outputDetectFrame = drawLines(roiExtractedImage, MINIMUM_LENGTH, MAXIMUM_LINE_GAP, LINE_DRAW_OPTION)
    
    #   DIsplay image side by side
    smallGray = cv2.resize(getGrayImage(frame), (0,0), fx=0.5, fy=0.5) 
    smallThreshold = cv2.resize(roiExtractedImage, (0,0), fx=0.5, fy=0.5)     
    combinedImage = np.hstack((smallGray, smallThreshold))

    cv2.imshow('Image',combinedImage)
    cv2.imshow('smallGray',smallGray)

    cv2.imshow('smallThreshold',smallThreshold)


    #   Publish Image
    # outputframe = CvBridge().cv2_to_imgmsg(combinedImage, encoding="passthrough")
    # LaneDetectionNode.publish(outputframe)

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

