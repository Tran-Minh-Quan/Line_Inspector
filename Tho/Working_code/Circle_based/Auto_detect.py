import numpy as np
import argparse
import cv2
import time
import math


def nothing():
    pass


# Example usage: python3 Detect_circle.py --image ./Changed_data/20.jpg
# For Pycharm: Edit Configuration -> Parameters -> --image ./Changed_data/20.jpg
img = cv2.imread("./Changed_data/20.jpg")
start_time = time.time()
Canny_low = 0
Canny_high = 2000
Canny_param = Canny_high
start_time = time.time()
RM_left = Canny_low
RM_right = Canny_high
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
output = img.copy()
step = 1
while RM_left < RM_right:
    Canny_param = math.floor((RM_right+RM_left)/2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=Canny_param, param2=10, minRadius=0, maxRadius=0)
    print(step)
    step += 1
    if circles is None:
        RM_right = Canny_param
    else:
        RM_left = Canny_param + 1
    # if circles.size > 3:
    #     RM_right = Canny_param
    # else:
    #     RM_left = Canny_param + 1
Canny_param = RM_left - 1
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                           param1=Canny_param, param2=30, minRadius=0, maxRadius=0)
circles_round = np.round(circles[0, :]).astype("int")
for (x, y, r) in circles_round:
    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    end_time = time.time()
    print(end_time-start_time)
    cv2.imshow("Window 1", np.hstack([img, output]))
    cv2.waitKey(2000)

# while True:
#     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
#                                param1=Canny_param, param2=30, minRadius=0, maxRadius=0)
#     if circles is not None:
#         circles_round = np.round(circles[0, :]).astype("int")
#         for (x, y, r) in circles_round:
#             cv2.circle(output, (x, y), r, (0, 255, 0), 4)
#             cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#         cv2.imshow("Window 1", np.hstack([img, output]))
#         if circles.size is not 3:
#             if Canny_param < Canny_high:
#                 Canny_param += Canny_step
#         else:
#             stop_time = time.time()
#             print("{}".format(stop_time - start_time))
#             print("R = {} pixel".format(circles[0][0][2]))
#             break
#     else:
#         if Canny_param >= Canny_low + Canny_step:
#             Canny_param -= Canny_step
#         else:
#             pass
#         cv2.imshow("Window 1", np.hstack([img, img]))
#     print(Canny_param)
#     k = cv2.waitKey(10) & 0xFF
#     if k == 27:
#         break
