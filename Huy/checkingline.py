import cv2
import numpy as np
from matplotlib import pyplot as plt
def checkingLine(image,normalValue=40,offset=5):
    image = cv2.imread(image)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    track_window = (0, 0,image.shape[1], image.shape[0])
    frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    frame1 = frame[75:95, 220:420].copy()
    template = cv2.imread('doanday2.jpg', 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(frame1, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w - 10, top_left[1] + h)
    frame2 = frame1[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]].copy()
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.adaptiveThreshold(frame2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY_INV, 897, 27)
    opening1 = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)
    ret, track_window = cv2.CamShift(opening, track_window, term_crit)
    pts = np.int0(cv2.boxPoints(ret))
    y1 = pts[0][0]
    y2 = pts[1][0]
    y3 = pts[2][0]
    y4 = pts[3][0]
    if (not (y1 == 0 and y2 == 0 and y3 == 0 and y4 == 0)) or (max(y1, y2, y3, y4) >= image.shape[1] - 30) or (
            min(y1, y2, y3, y4) <= 30):
        opening[:, 0:min(y1, y2, y3, y4) - 10] = 0
        opening[:, max(y1, y2, y3, y4) + 10:image.shape[1]] = 0
        resu = opening[:, max(min(y1, y2, y3, y4) - 10, 0):min(max(y1, y2, y3, y4) + 10, image.shape[1])].copy()
        edges = cv2.Canny(opening, 50, 200, apertureSize=3)
        feat = np.sum(edges / 255)
        tl=top_left
        br=bottom_right
        if ((feat <= (normalValue - offset)) or (feat >= (normalValue + offset))):
            return 0,top_left,bottom_right
        else:
            return 1, top_left,bottom_right
        cv2.waitKey(10)


cv2.destroyAllWindows()

print(checkingLine(image='line1.jpg', normalValue=40,offset=5))





