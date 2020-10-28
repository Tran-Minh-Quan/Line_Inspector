import cv2
import numpy as np
from distance_calc_lib import edge_based


def nothing():
    pass


img = cv2.imread("Clamp/clamp_30_1.jpg")
cv2.namedWindow("Result")
cv2.createTrackbar("Canny_1", "Result", 391, 1000, nothing)
cv2.createTrackbar("Canny_2", "Result", 664, 2000, nothing)
while True:
    Canny_1 = cv2.getTrackbarPos("Canny_1", "Result")
    Canny_2 = cv2.getTrackbarPos("Canny_2", "Result")
    distance, img_out, _ = edge_based(img, (100, 0), (450, 500), 0, Canny_1, Canny_2, 764, 7)
    cv2.imshow("Result", img_out)
    print(distance/10)
    k = cv2.waitKey(100)
    if k == 27:
        break