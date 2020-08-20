import imutils
import cv2
import time
import dlib
import os
import glob
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from Tho import Distance_functions as Tho

def pyramid(image, scale=1, minSize=(30, 30)):
  while True:
    w = int(image.shape[1] / scale)
    image = imutils.resize(image, width=w)

    if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
      break
    return image


def combineImage(image1, image2, image, scaleim=1):
    image1 = pyramid(image=image1, scale=scaleim)
    image2 = pyramid(image=image2, scale=scaleim)
    image = pyramid(image=image, scale=scaleim)
    width = int(image.shape[1])
    half_width = int(image.shape[1] / 2)
    width1 = int(image1.shape[1])
    width2 = int(image2.shape[1])
    image[:, 0:half_width] = image1[:, 0:width1].copy()
    image[:, half_width:width] = image2[:, 0:width2].copy()
    return image


def Demo(scaleim, cam, vector_w, normalValue=40, offset=5):
    Distance = 1000
    ob_in_area = 0
    sdetect = 0
    frball = 0
    frta = 0
    frkhoa = 0
    nhan_dang = 0
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    track_window = (0, 0, 95 - 75, 420 - 220)
    Image_show = cv2.imread('An/imagezero.jpg')
    out = cv2.VideoWriter('An/result_backup.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 480))

    net = cv2.dnn.readNetFromDarknet("Model/YOLOv3/yolov3.cfg", "Model/YOLOv3/yolov3_best.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    classes = []
    with open("Model/YOLOv3/obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # tai anh/video
    start = time.time()
    cap = cv2.VideoCapture(cam)

    frame_id = 0
    while True:
        ok, frame = cap.read()
        frame_id += 1
        if not ok:
            break
        img = frame.copy()
        #gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        image = pyramid(img, scaleim)
        nearOb = -1

        height, width,channels = image.shape
        # nhandien
        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # hien_man_hinh
        detector_idxs = []
        confidences = []
        boxes = []
        for o in outs:
            for detection in o:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rec coord
                    x = int(center_x - w / 2)  # top left x
                    y = int(center_y - h / 2)  # top left y

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    detector_idxs.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0, 0.4)
        x=0
        y=0
        w=0
        h=0
        for i in range(len(boxes)):

            if i in indexes:
                nhan_dang = 1
                x,y,w,h = boxes[i]

                if (((detector_idxs[i] == 2) and (y <= 200)) or (y <= 95)):
                    ob_in_area = 1
                w_0 = vector_w[0][detector_idxs[i]]
                w_1 = vector_w[1][detector_idxs[i]]
                w_2 = vector_w[2][detector_idxs[i]]
                w_3 = vector_w[3][detector_idxs[i]]
                w_4 = vector_w[4][detector_idxs[i]]
                dis1 = 0
                dis2 = 0
                dis3 = 0
                frball = 0
                frta = 0
                frkhoa = 0
                if (detector_idxs[i] == 0):
                    dis1 = w_0 + w_1 * x + w_2 * y + w_3 * w + w_4 * h
                    frball += 1
                elif (detector_idxs[i] == 1):
                    dis2 = 1 / (w_0 + w_1 * y)
                    frta += 1
                else:
                    dis3 = w_0 + w_1 * y
                    frkhoa += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 250, 0), 2)
                cv2.putText(img, str(detector_idxs[i]) + '  ' + str(np.round(confidences[i] * 100, 2)) + '%', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 1, lineType=cv2.LINE_AA)

        frtotal = max(frball, frta, frkhoa)
        if (frtotal == 0):
            nearOb = -1
            Distance = 1000
        elif (frtotal == frball):
            nearOb = 0
            Distance = dis1
        elif (frtotal == frta):
            nearOb = 1
            Distance = dis2
        elif (frtotal == frkhoa):
            nearOb = 2
            Distance = dis3
        if nearOb == 0:
            NameOb = 'Marker Ball'
        elif nearOb == 1:
            NameOb = 'Ta Chong Rung'
        elif nearOb == 2:
            NameOb = 'Khoa do day'
        else:
            NameOb = 'Nothing'

        #Phan cua Tho
        top_left = [x,y]
        bottom_right = [x+w,y+h]
        Distance, contour, _ = Tho.distance_estimate(frame.copy(), top_left, bottom_right, 0.2, 200, 400, 14.7)
        Distance = Distance / 10
        cv2.imshow('contour', contour.copy())


        '''
        if top_left == [0,0] or bottom_right == [0,0]:
            Distance = 1000
        else:
            Distance,_,_ = Tho.distance_estimate(img, top_left, bottom_right, 0.2, 300, 900, 14.7)
            Distance = Distance/10
        '''

        if (Distance == 1000 or nhan_dang == 0 or np.round(Distance, 2) < 21):
            dis_string = str(np.round(Distance, 2))
            #dis_string = 'Unknow'
        else:
            dis_string = str(np.round(Distance, 2))
        end = time.time()
        cv2.putText(img, 'FPS:' + str(
            np.round(1 / (end - start))) + '   Name:' + NameOb + '     Distance:' + dis_string + '  cm', (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        start = time.time()

        frame1 = frame[75:95, 220:420].copy()
        cv2.rectangle(frame, (220, 75), (420, 95), (0, 255, 0), 2)
        frame2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        th = cv2.adaptiveThreshold(frame2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY_INV, 51, 4)
        opening1 = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        opening = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)
        ret, track_window = cv2.CamShift(opening, track_window, term_crit)
        pts = np.int0(cv2.boxPoints(ret))
        y1 = pts[0][0]
        y2 = pts[1][0]
        y3 = pts[2][0]
        y4 = pts[3][0]
        if (not (y1 == 0 and y2 == 0 and y3 == 0 and y4 == 0)) or (max(y1, y2, y3, y4) >= 150) or (
                min(y1, y2, y3, y4) <= 50):
            opening[:, 0:min(y1, y2, y3, y4) - 10] = 0
            opening[:, max(y1, y2, y3, y4) + 10:200] = 0
            resu = opening[:, max(min(y1, y2, y3, y4) - 10, 0):min(max(y1, y2, y3, y4) + 10, 200)].copy()
            edges = cv2.Canny(opening, 50, 200, apertureSize=3)
            feat = np.sum(edges / 255)
            #print(feat)
            if sdetect == 1:
                if ob_in_area == 1:
                    cv2.putText(frame, 'Obstacle', (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2,
                                lineType=cv2.LINE_AA)
                    cv2.putText(img, 'Lightning rods: Unknow', (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                                1, lineType=cv2.LINE_AA)
                    cv2.rectangle(frame, (220, 75), (420, 95), (255, 0, 0), 2)
                elif ((feat <= 37) or (feat >= 43)):
                    cv2.putText(frame, 'Error', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                                lineType=cv2.LINE_AA)
                    cv2.putText(img, 'Lightning rods: Error', (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                                1, lineType=cv2.LINE_AA)
                    cv2.rectangle(frame, (220, 75), (420, 95), (0, 0, 255), 2)
                else:
                    cv2.putText(frame, 'Normal', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                                lineType=cv2.LINE_AA)
                    cv2.putText(img, 'Lightning rods: Normal', (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                                1, lineType=cv2.LINE_AA)
                    cv2.rectangle(frame, (220, 75), (420, 95), (0, 255, 0), 2)


        Image_show = combineImage(img, frame, Image_show, scaleim=1)
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', Image_show)
        Distance = 1000
        nhan_dang = 0
        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break
        elif (k == 115):
            sdetect = 1
        elif (k == 99):
            ob_in_area = 0
            frball = 0
            frta = 0
            frkhoa = 0
        out.write(Image_show)
    cap.release()
    cv2.destroyAllWindows()









