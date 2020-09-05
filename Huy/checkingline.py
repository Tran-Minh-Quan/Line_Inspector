import cv2
import numpy as np
import os


class Line:
    def __init__(self,frame,ROI):
        self.top_left_x = ROI[0]
        self.top_left_y = ROI[1]
        self.bottom_right_x = ROI[2]
        self.bottom_right_y = ROI[3]
        self.ROI_image = frame[ROI[1]:ROI[3], ROI[0]:ROI[2]].copy()

    def detect_error(self,template_dir, min_thres = 36, max_thres = 48):
        frame1 = cv2.cvtColor(self.ROI_image,cv2.COLOR_BGR2GRAY)
        template = cv2.imread(template_dir, 0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(frame1, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (220+max_loc[0],75+max_loc[1])
        bottom_right = (top_left[0] + w, top_left[1] + h)
        self.top_left_x = self.top_left_x + max_loc[0]
        self.top_left_y = self.top_left_y + max_loc[1]
        self.bottom_right_x = self.top_left_x + w
        self.bottom_right_y = self.top_left_y + h
        self.ROI_image = frame1[max_loc[1]:h, max_loc[0]:max_loc[0]+w]
        opening_kernel = np.ones((3, 3), np.uint8)

        th = cv2.adaptiveThreshold(self.ROI_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY_INV,241,15)
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, opening_kernel)
        #cv2.imshow('frame2',frame2)
        #edges = cv2.Canny(opening, 50, 200, apertureSize=3)
        dilation_kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(opening, dilation_kernel)
        edge_image = dilation - opening
        self.edge_num = np.sum(edge_image) // 255
        #cv2.imshow('edges',edges)
        if ((self.edge_num < min_thres) or (self.edge_num > max_thres)):
            self.isError = 1
        else:
            self.isError = 0



#test hàm checkingLine
if __name__ == "__main__":
    cap=cv2.VideoCapture('video_test_day.avi')

    N_bad=[]
    while(True):
        ok,frame=cap.read()
        if not ok:
          break
        else:
          line = Line(frame=frame, ROI=[220, 75, 420, 95])
          line.detect_error(template_dir='D:\\WON\\DO_AN\\Code\\Huy\\template.jpg',min_thres=36, max_thres=48)
          #print(checkOk)
          if not line.isError:
            cv2.putText(frame, 'Normal', (line.top_left_x-130,line.top_left_y+20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, (line.top_left_x, line.top_left_y), (line.bottom_right_x, line.bottom_right_y), (0,255,0), 2)
          else:
            cv2.putText(frame, 'Error', (line.top_left_x-130,line.top_left_y+20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, (line.top_left_x, line.top_left_y), (line.bottom_right_x, line.bottom_right_y), (0,0,255), 2)
            N_bad.append(line.isError)
        cv2.imshow('result',frame)
        k = cv2.waitKey(50) & 0xff
        if (k == 27):
          break
        elif (k==115):
          start=1
    cap.release()
    cv2.destroyAllWindows()






