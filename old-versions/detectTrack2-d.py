# tracking, only cars were keep on adding (using mid-values)
# no removal from array

import cv2
import numpy as np
import copy

net = cv2.dnn.readNet("../yolov3.weights", "../yolov3.cfg")
classes = []
with open("../coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('../videos/highway.mp4')
# cap=cv2.VideoCapture('ball_tracking_example.mp4')

prev_frame=[]
number=0
while True:
    _,img=cap.read()
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
            cx,cy = (2*x + w)/2  ,  (2*y + h)/2
            curr_frame.append([x,y,w,h,cx,cy,label])

    # object tracking 
    curr , prev=copy.deepcopy(curr_frame) , copy.deepcopy(prev_frame)
    display=[]

    for i in range(min([len(prev_frame),len(curr_frame)])):
        small=9999
        for curr_ind,curr_obj in enumerate(curr):
            x2 , y2 = curr_obj[4] , curr_obj[5]
            for prev_ind,prev_obj in enumerate(prev):
                x1 , y1 = prev_obj[0] , prev_obj[1]
                ans=((x2-x1)**2+(y2-y1)**2)**(1/2)
                if ans<small:
                    small = ans
                    ind = prev_obj[2]
                    pop1 , pop2 = curr_ind , prev_ind
                    new_list = [x2 , y2 , ind]
                    disp=curr_obj
        if small < 100:
            display.append([disp,ind])
            curr.pop(pop1)
            prev.pop(pop2)
            change.append(new_list)
        else:
            break
                
    for i in display:
        color=colors[i[1]%75]
        x,y,w,h,label = i[0][0] , i[0][1] , i[0][2] , i[0][3] , i[0][6] 
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text=' '+str(i[1])
        cv2.putText(img, text, (x, y + 30), font, 3, color, 2)

    for i in curr:
        number=number+1
        color=colors[number%75]
        prev_frame.append([i[4],i[5],number])
        x,y,w,h,label = i[0] , i[1] , i[2] , i[3] , i[6] 

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text=' '+str(number)
        cv2.putText(img, text, (x, y + 30), font, 3, color, 2)




    if number==0:
        for i in curr_frame:
            number=number+1
            color=colors[number%75]
            x,y,w,h,cx,cy,label = i
            prev_frame.append([cx,cy,number])

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text=' '+str(number)
            cv2.putText(img, text, (x, y + 30), font, 3, color, 2)




    print(number , len(prev_frame),len(curr_frame))
    print()
    for ch in change:
        find=ch[2]
        for rr,ob in enumerate(prev_frame):
            if ob[2]==find:
                prev_frame[rr][0],prev_frame[rr][1]=ch[0],ch[1]
                break


    cv2.imshow("Image", img)
    key=cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()