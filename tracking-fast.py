# adding and removal of cars after tracking (using iou)
# detect only vehicles
# detection using tf

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import pathlib
import cv2
import imutils
import time
from sklearn.metrics import pairwise
from imutils.video import FPS
import copy
import pathlib



from utils import ops as utils_ops
from utils import label_map_util



utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
PATH_TO_LABELS = '../bigdata/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)




model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09'
model_dir =  "../bigdata/models/" + model_name + "/saved_model"
detection_model = tf.saved_model.load(str(model_dir))
detection_model = detection_model.signatures['serving_default']



# print(category_index)
colors = np.random.uniform(0, 255, size=(len(category_index), 3))
font = cv2.FONT_HERSHEY_SIMPLEX

print(detection_model.inputs)
print(detection_model.output_dtypes)
print(detection_model.output_shapes)

  


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




def visualize(output_dict,image_np,height,width):
  class_ids = []
  confidences = []
  boxes = []
  num = output_dict['num_detections']
  for ind in range(num):
    scr = output_dict['detection_scores'][ind]
    classId = output_dict['detection_classes'][ind] 
    box = output_dict['detection_boxes'][ind]

    ymin, xmin, ymax, xmax = box
    confidences.append(float(scr))
    class_ids.append(classId)
    boxes.append([int(xmin*width) , int(ymin*height) , int((xmax-xmin)*width) , int((ymax-ymin)*height)])

  indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
  # if len(boxes) != len(indexes):
  #   print(indexes,boxes , confidences,class_ids)
  for j in indexes:
    i = j[0]
    x, y, w, h = boxes[i]
    label = category_index[class_ids[i]]['name']
    color = colors[i]
    cv2.rectangle(image_np, (x, y), (x + w, y + h), color, 3)
    cv2.putText(image_np, label, (x, y - 5), font, 3, color, 3)
  # return image_np



def tracking(indexesCars , boxesCars , image_np):
    global prev_frame , number
    change=[]
    curr_frame=[]
    for j in indexesCars:
        i = j[0]
        x, y, w, h = boxesCars[i]
        label = "vehicle"
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
                if ans > small:
                    small = ans
                    ll1,ll2=l1,l2
                    ind = prev_obj[4]
                    chct = prev_obj[5]
                    pop1 , pop2 = curr_ind , prev_ind
                    new_list = [ curr_obj[0], curr_obj[1], curr_obj[2], curr_obj[3] , ind ]
                    disp=curr_obj
        # print(small,aa,bb,cc,dd)
        # print(curr,prev)
        # print(small,ll1,ll2)
        # print(len(curr_frame) , len(prev_frame) , len(curr) , len(prev))
        # print(min([len(prev_frame),len(curr_frame)]))
        if small > 0.45:                   # decrease this if objects are small and their iou can change to a greater extent
            display.append([disp,ind,chct])
            curr.pop(pop1)
            prev.pop(pop2)
            change.append(new_list)
        else:
            break
        # print(len(change),len(display))

    # print('display curr',display,curr)       
    for i in display:
        color=colors[i[1]%75]
        x1,y1,x2,y2,label = i[0][0] , i[0][1] , i[0][2] , i[0][3] , i[0][4] 
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        text=' '+str(i[1]) 
        cv2.putText(image_np, text, (x1, y1 + 30), font, 3, color, 2)

    for i in curr:
        number=number+1
        color=colors[number%75]
        prev_frame.append([i[0], i[1], i[2], i[3], number,0])
        x1, y1, x2, y2, label = i[0] , i[1] , i[2] , i[3] , i[4] 

        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        text=' '+str(number)
        cv2.putText(image_np, text, (x1, y1 + 30), font, 3, color, 2)


    if number==0:
        for i in curr_frame:
            number=number+1
            color=colors[number%75]
            xx1, yy1, xx2, yy2, label = i
            prev_frame.append([xx1, yy1, xx2, yy2, number, 0])

            cv2.rectangle(image_np, (xx1, yy1), (xx2, yy2), color, 2)
            text=' '+str(number)
            cv2.putText(image_np, text, (xx1, yy1 + 30), font, 3, color, 2)


    # print(number , len(prev_frame),len(curr_frame))
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
    # print("prev_frame",prev_frame)
    # print(index_note)
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
    # print('after pop',prev_frame)
    return image_np



def show_inference(model, image_path):
    image_np = np.array(image_path)
    height,width,channel = image_np.shape

    image = np.asarray(image_np)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]

    # output_dict is a dict  with keys detection_classes , num_detections , detection_boxes(4 coordinates of each box) , detection_scores for 100 boxes
    output_dict = model(input_tensor)


    # num_detections gives number of objects in current frame
    num_detections = int(output_dict.pop('num_detections'))
    # output_dict is a dict  with keys detection_classes , detection_boxes(4 coordinates of each box) , detection_scores for num_detections boxes
    output_dict = {key:value[0, :num_detections].numpy() 
                  for key,value in output_dict.items()}
    # adding num_detections that was earlier popped out
    output_dict['num_detections'] = num_detections
    # converting all values in detection_classes as ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    # print(5,output_dict)

    confidencesCars = []
    boxesCars = []

    num = output_dict['num_detections']
    for ind in range(num):
        classId = output_dict['detection_classes'][ind] 
        if classId==2 or classId==3 or classId==4 or classId==6 or classId==8:
            scr = output_dict['detection_scores'][ind]
            box = output_dict['detection_boxes'][ind]
            ymin, xmin, ymax, xmax = box
            w = (xmax - xmin) * width
            h = (ymax - ymin) * height
            if (w*h >=800):
                boxesCars.append([int(xmin*width) , int(ymin*height) , int(w) , int(h)])
                confidencesCars.append(float(scr))

    indexesCars = cv2.dnn.NMSBoxes(boxesCars, confidencesCars, 0.5, 0.4)

    image_np = tracking(indexesCars , boxesCars , image_np)


    # visualize(output_dict,image_np,height,width)
    cv2.imshow("version", image_np)









# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture('../videos/b.mp4')
time.sleep(2.0)

cap.set(1,100)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out1 = cv2.VideoWriter('i.avi', fourcc, 3.0, (int(cap.get(3)),int(cap.get(4))))

fps = FPS().start()
ctt = 0
number=0
prev_frame = []
while True:
    _,frame=cap.read()
    height, width, channels = frame.shape

    show_inference(detection_model, frame)

    cv2.imshow("original", frame)
    # out1.write(frame)
    fps.update()

    key=cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break
        
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
# out1.release()
cv2.destroyAllWindows() 



