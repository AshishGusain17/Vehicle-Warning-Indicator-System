# adding and removal of cars after tracking (using iou)
# detect only vehicles

import cv2
import numpy as np
import copy
from imutils.video import FPS

net = cv2.dnn.readNet("../../yolov3.weights", "../../yolov3.cfg")
classes = []
with open("../../coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    # print(interArea, float(boxAArea + boxBArea - interArea))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('../../videos/a.mp4')
cap.set(1,0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out1 = cv2.VideoWriter('i.avi', fourcc, 3.0, (int(cap.get(3)),int(cap.get(4))))
fps = FPS().start()

prev_frame=[]
number=0
cot=0
while True:
    _,img=cap.read()
    cot=cot+1
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            # print(classes[class_id])
            if (classes[class_id]=='person' or classes[class_id]=='bicycle' or classes[class_id]=='car' or classes[class_id]=='motorbike'  or classes[class_id]=='bus'  or classes[class_id] =='truck'):

	            confidence = scores[class_id]
	            if confidence > 0.5:
	                # Object detected
	                center_x = int(detection[0] * width)
	                center_y = int(detection[1] * height)

	                w = int(detection[2] * width)
	                h = int(detection[3] * height)

	                # Rectangle coordinates
	                x = int(center_x - w / 2)
	                y = int(center_y - h / 2)
	                if (w*h >=1800):
		                boxes.append([x, y, w, h])
		                confidences.append(float(confidence))
		                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN


    change=[]
    curr_frame=[]

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # color = colors[i]
            # cx,cy = (2*x + w)/2  ,  (2*y + h)/2
            curr_frame.append([x,y,x+w,y+h,label])

    # object tracking 
    curr , prev=copy.deepcopy(curr_frame) , copy.deepcopy(prev_frame)
    display=[]
    ll1,ll2,l1,l2=[],[],[],[]
    ans=0
    for i in range(max([len(prev_frame),len(curr_frame)])):
        small=0
        for curr_ind,curr_obj in enumerate(curr):
            # x2 , y2 = curr_obj[4] , curr_obj[5]
            l1 = [ curr_obj[0], curr_obj[1], curr_obj[2], curr_obj[3] ]
            for prev_ind,prev_obj in enumerate(prev):
                # x1 , y1 = prev_obj[0] , prev_obj[1]
                l2 = [ prev_obj[0], prev_obj[1], prev_obj[2], prev_obj[3] ]
                ans = iou( l1 , l2 )
                # ans=((x2-x1)**2+(y2-y1)**2)**(1/2)
                if ans > small:
                    small = ans
                    ll1,ll2=l1,l2
                    ind = prev_obj[4]
                    chct = prev_obj[5]
                    # aa,bb,cc,dd=x1,y1,x2,y2
                    pop1 , pop2 = curr_ind , prev_ind
                    new_list = [ curr_obj[0], curr_obj[1], curr_obj[2], curr_obj[3] , ind ]
                    disp=curr_obj
        # print(small,aa,bb,cc,dd)
        print(curr,prev)
        print(small,ll1,ll2)
        print(len(curr_frame) , len(prev_frame) , len(curr) , len(prev))
        # print(min([len(prev_frame),len(curr_frame)]))
        if small > 0.45:                   # decrease this if objects are small and their iou can change to a greater extent
            display.append([disp,ind,chct])
            curr.pop(pop1)
            prev.pop(pop2)
            change.append(new_list)
        else:
            break
        print(len(change),len(display))

    print('display curr',display,curr)       
    for i in display:
        color=colors[i[1]%75]
        x1,y1,x2,y2,label = i[0][0] , i[0][1] , i[0][2] , i[0][3] , i[0][4] 
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text=' '+str(i[1]) 
        cv2.putText(img, text, (x1, y1 + 30), font, 3, color, 2)

    for i in curr:
        number=number+1
        color=colors[number%75]
        prev_frame.append([i[0], i[1], i[2], i[3], number,0])
        x1, y1, x2, y2, label = i[0] , i[1] , i[2] , i[3] , i[4] 

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text=' '+str(number)
        cv2.putText(img, text, (x1, y1 + 30), font, 3, color, 2)




    if number==0:
        for i in curr_frame:
            number=number+1
            color=colors[number%75]
            xx1, yy1, xx2, yy2, label = i
            prev_frame.append([xx1, yy1, xx2, yy2, number, 0])

            cv2.rectangle(img, (xx1, yy1), (xx2, yy2), color, 2)
            text=' '+str(number)
            cv2.putText(img, text, (xx1, yy1 + 30), font, 3, color, 2)




    # print(number , len(prev_frame),len(curr_frame))
    print()
    for ch in change:
        find=ch[4]
        for rr,ob in enumerate(prev_frame):
            if ob[4]==find:
                prev_frame[rr][0], prev_frame[rr][1], prev_frame[rr][2], prev_frame[rr][3], prev_frame[rr][5] = ch[0], ch[1], ch[2], ch[3], 0
                break


    index_note=[]
    for rr,ob in enumerate(prev_frame):
        prev_frame[rr][5]+=1
        if prev_frame[rr][5]>=6:
            index_note.append(rr)
    print("prev_frame",prev_frame)
    print(index_note)
    lll=[]
    for rr,ob in enumerate(prev_frame):
    	flag=0
    	for j in index_note:
    		if j==rr:
    			flag=1
    			break
    	if flag==0:
    		lll.append(ob)
    prev_frame=lll
    print('after pop',prev_frame)

    cv2.imshow("version", img)
    # out1.write(img)
    key=cv2.waitKey(1)
    fps.update()
    if key & 0xFF == ord("q"):
        break
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
# out1.release()
cv2.destroyAllWindows()