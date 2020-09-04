import cv2
import numpy as np
import os
#test git push
def checkingLine(image,template_dir,normalValue=42,offset=6):
    frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    frame1 = frame[75:95, 220:420]
    template = cv2.imread(template_dir, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(frame1, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = (220+max_loc[0],75+max_loc[1])
    bottom_right = (top_left[0] + w, top_left[1] + h)


    frame2 = frame[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]].copy()

    opening_kernel = np.ones((3, 3), np.uint8)
    th = cv2.adaptiveThreshold(frame2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY_INV,241,15)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, opening_kernel)
    #cv2.imshow('frame2',frame2)
    #edges = cv2.Canny(opening, 50, 200, apertureSize=3)
    dilation_kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(opening, dilation_kernel)
    edges = dilation - opening
    feat = np.sum(edges) // 255
    #cv2.imshow('edges',edges)

    if ((feat <= (normalValue - offset)) or (feat >= (normalValue + offset))):
        return 1,top_left,bottom_right
    else:
        return 0,top_left,bottom_right


#test hàm checkingLine
if __name__ == "__main__":
    cap=cv2.VideoCapture('video_test_day.avi')

    N_bad=[]
    while(True):
        ok,frame=cap.read()
        if not ok:
          break
        else:
          checkOk,top_left, bottom_right = checkingLine(image=frame,template_dir='template.jpg',normalValue=42,offset=6)
          #print(checkOk)
          if(checkOk==0):
            cv2.putText(frame, 'Normal', (top_left[0]-130,top_left[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (0,255,0), 2)
          else:
            cv2.putText(frame, 'Error', (top_left[0]-130,top_left[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (0,0,255), 2)
            N_bad.append(checkOk)
        cv2.imshow('result',frame)
        k = cv2.waitKey(50) & 0xff
        if (k == 27):
          break
        elif (k==115):
          start=1
    cap.release()
    cv2.destroyAllWindows()






