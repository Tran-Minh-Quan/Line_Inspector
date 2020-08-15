import cv2
import numpy as np
import time
net = cv2.dnn.readNetFromDarknet( "yolov3.cfg","yolov3_best.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#tai anh/video
#cap=cv2.VideoCapture(0)
#start_time=time.time()
#frame_id=0
#while True:
    #_,frame=cap.read()
    #frame_id+=1
a = 'D:/WON/DO_AN/Training/test/a(1).jpg'
img=cv2.imread(a)
#img=cv2.resize(img,None,fx=2,fy=2)
#height,width,channels=frame.shape
#nhandien
#blob=cv2.dnn.blobFromImage(frame,0.00392,(416,416),(0,0,0),True,crop=False)
blob=cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)

net.setInput(blob)
outs=net.forward(output_layers)

#hien_man_hinh
class_ids=[]
confidences=[]
boxes=[]
for out in outs:
    for detection in out:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]
        if confidence >0.5:
            center_x=int(detection[0]*width)
            center_y=int(detection[1]*height)
            w= int(detection[2]*width)
            h=int(detection[3]*height)


            #Rec coord
            x=int(center_x-w/2) #top left x
            y= int(center_y-h/2)#top left y

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
number_object_detected =print("so object nhan dang duoc la : " +str(len(indexes)))

for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label=str(classes[class_ids[i]])
        conf=confidences[i]
        cv2. rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,label,(x+20,y+30),cv2.FONT_HERSHEY_PLAIN,1,(0,10,10),3)
        print(label + str(boxes[i])+str(conf))








#end_time=time.time()-start_time
#fps=frame_id/end_time
#cv2.putText(frame,"FPS: "+str(fps),(10,30),cv2.FONT_HERSHEY_PLAIN,3,(230,230,230),1)
#cv2.imshow("thanh_cong",frame)
cv2.imshow("thanh_cong", img)
cv2.imwrite(a + '_result.jpg',img)
key=cv2.waitKey(-1)
#if key==27:
    #break

cv2.destroyAllWindows()
'''
import cv2 as cv

net = cv.dnn_DetectionModel('yolov4.cfg', 'yolov4.weights')
net.setInputSize(608, 608)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)

frame = cv.imread('lena.jpg')

with open('coco.names', 'rt') as f:
    names = f.read().rstrip('\n').split('\n')

classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
    label = '%.2f' % confidence
    label = '%s: %s' % (names[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    left, top, width, height = box
    top = max(top, labelSize[1])
    cv.rectangle(frame, box, color=(0, 255, 0), thickness=3)
    cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

cv.imshow('out', frame)
'''