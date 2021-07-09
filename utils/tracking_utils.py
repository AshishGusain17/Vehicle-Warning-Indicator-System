import numpy as np
import os
import sys
import tensorflow as tf
from imutils.video import VideoStream
import cv2
import imutils
import time
from imutils.video import FPS
from sklearn.metrics import pairwise
import copy
import pathlib
from collections import defaultdict



colors = np.random.uniform(0, 255, size=(100, 3))
font = cv2.FONT_HERSHEY_SIMPLEX




def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # print(interArea, float(boxAArea + boxBArea - interArea))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou



def tracking(indexesCars , boxesCars , image_np , prev_frame , number):
    change=[]
    curr_frame=[]
    for j in indexesCars:
        i = j[0]
        x, y, w, h = boxesCars[i]
        label = "vehicle"
        curr_frame.append([x,y,x+w,y+h,label])

    # object tracking 
    curr , prev=copy.deepcopy(curr_frame) , copy.deepcopy(prev_frame)
    tr_frame = copy.deepcopy(image_np)
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
                if ans > small:
                    small = ans
                    ll1,ll2=l1,l2
                    ind = prev_obj[4]
                    chct = prev_obj[5]
                    pop1 , pop2 = curr_ind , prev_ind
                    new_list = [ curr_obj[0], curr_obj[1], curr_obj[2], curr_obj[3] , ind ]
                    disp=curr_obj

        if small > 0.45:                   # decrease this if objects are small and their iou can change to a greater extent
            display.append([disp,ind,chct])
            curr.pop(pop1)
            prev.pop(pop2)
            change.append(new_list)
        else:
            break
    
    for i in display:
        color=colors[i[1]%75]
        x1,y1,x2,y2,label = i[0][0] , i[0][1] , i[0][2] , i[0][3] , i[0][4] 
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

        cv2.rectangle(tr_frame, (x1, y1), (x2, y2), color, 2)
        text=' '+str(i[1]) 
        cv2.putText(tr_frame, text, (x1, y1 + 30), font, 1.2, color, 2)

    for i in curr:
        number=number+1
        color=colors[number%75]
        prev_frame.append([i[0], i[1], i[2], i[3], number,0])
        x1, y1, x2, y2, label = i[0] , i[1] , i[2] , i[3] , i[4] 

        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

        cv2.rectangle(tr_frame, (x1, y1), (x2, y2), color, 2)
        text=' '+str(number)
        cv2.putText(tr_frame, text, (x1, y1 + 30), font, 1.2, color, 2)


    if number==0:
        for i in curr_frame:
            number=number+1
            color=colors[number%75]
            xx1, yy1, xx2, yy2, label = i
            prev_frame.append([xx1, yy1, xx2, yy2, number, 0])
            cv2.rectangle(image_np, (xx1, yy1), (xx2, yy2), color, 2)

            cv2.rectangle(tr_frame, (xx1, yy1), (xx2, yy2), color, 2)
            text=' '+str(number)
            cv2.putText(tr_frame, text, (xx1, yy1 + 30), font, 1.2, color, 2)



    for ch in change:
        find=ch[4]
        for rr,ob in enumerate(prev_frame):
            if ob[4]==find:
                prev_frame[rr][0], prev_frame[rr][1], prev_frame[rr][2], prev_frame[rr][3], prev_frame[rr][5] = ch[0], ch[1], ch[2], ch[3], 0
                break


    index_note=[]
    for rr,ob in enumerate(prev_frame):
        prev_frame[rr][5]+=1
        if prev_frame[rr][5]>=20:
            index_note.append(rr)


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

    cv2.imshow("tracking", tr_frame)
    return  image_np , prev_frame , number




