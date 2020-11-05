from Quan.QuanLib import Yolo
import cv2

# cap = cv2.VideoCapture(r'D:\WON\DO_AN\Data\Line\Video\line_3.avi')
cap = cv2.VideoCapture(1)

cap.set(4, int(480))
cap.set(3, int(640))
model = Yolo(weights_dir=r'D:\WON\DO_AN\Data\Line\training_result\yolov4-custom_last.weights',
             cfg_dir=r'D:\WON\DO_AN\Data\Line\training_result\yolov4-custom.cfg',
             names_dir=r'D:\WON\DO_AN\Data\Line\training_result\obj.names')
while True:
    ok, frame = cap.read()
    if ok:
        objects = model.detect(frame[210:380, 0:640])
        if objects:
            for obj in objects:
                top_left = list(obj['top left'])
                top_left[1] = 210
                top_left = tuple(top_left)
                bottom_right = list(obj['bottom right'])
                bottom_right[1] = 380
                bottom_right = tuple(bottom_right)
                if obj['name'] == 'normal':
                    cv2.rectangle(frame, top_left, bottom_right, (55, 100, 0), 2)
                    cv2.putText(frame, 'normal ' + str(obj['confidence']) + '%', (top_left[0], top_left[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (55, 100, 0), 2, lineType=cv2.LINE_AA)
                else:
                    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
                    cv2.putText(frame, 'error ' + str(obj['confidence']) + '%', (top_left[0], top_left[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        else:
            top_left = (0, 210)
            bottom_right = (640, 380)
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
            cv2.putText(frame, '0%', (top_left[0] + 130, top_left[1] - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.imshow('Line damage detect', frame)
        cv2.waitKey(1)
    else:
        break

