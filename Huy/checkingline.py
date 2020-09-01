import cv2
import numpy as np
def checkingLine(image,normalValue=40,offset=5):
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    track_window = (0, 0,image.shape[1], image.shape[0])
    frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    frame1 = frame[75:95, 220:420].copy()
    template = cv2.imread('template.jpg', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(frame1, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w - 10, top_left[1] + h)


    frame2 = frame1[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]].copy()
    kernel = np.ones((3, 3), np.uint8)
    opening_kernel = np.ones((3, 5), np.uint8)
    th = cv2.adaptiveThreshold(frame2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY_INV,61,7)
    opening1 = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, opening_kernel )
    ret, track_window = cv2.CamShift(opening, track_window, term_crit)
    cv2.imshow('frame2',frame2)
    edges = cv2.Canny(opening, 50, 200, apertureSize=3)
    feat = np.sum(edges / 255)
    cv2.imshow('edges',edges)

    tl=top_left
    br=bottom_right
    if ((feat <= (normalValue - offset)) or (feat >= (normalValue + offset))):
        return 1
    else:
        return 0
cv2.destroyAllWindows()

#test hàm checkingLine
cap=cv2.VideoCapture('video_test_day.avi')

N_bad=[]
while(True):
    ok,frame=cap.read()
    if not ok:
      break
    else:
      checkOk= checkingLine(image=frame,normalValue=40,offset=5)
      print(checkOk)
      if(checkOk==0):
        cv2.putText(frame, 'Normal', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (220,75), (420,95), (0,255,0), 2)
      else:
        cv2.putText(frame, 'Error', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (220,75), (420,95), (0,0,255), 2)
        N_bad.append(checkOk)
    cv2.imshow('result',frame)
    cv2.waitKey(50)
    k = cv2.waitKey(5) & 0xff
    if (k == 27):
      break
    elif (k==115):
      start=1
cap.release()
cv2.destroyAllWindows()




