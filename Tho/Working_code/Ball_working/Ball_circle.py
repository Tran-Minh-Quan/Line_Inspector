import cv2
import numpy as np
from distance_calc_lib import CircleBasedObjectDistance


def nothing():
    pass


CircleBasedObjectDistance_1 = CircleBasedObjectDistance(764)
img = cv2.imread("Ball/ball_39_1.jpg")
red_channel = img[:, :, 2]
green_channel = img[:, :, 1]
blue_channel = img[:, :, 0]
cv2.namedWindow("Eval image")
cv2.namedWindow("Result")
cv2.createTrackbar("Green_coefficient", "Eval image", 500, 4000, nothing)  # 112 #500
cv2.createTrackbar("Blue_coefficient", "Eval image", 1038, 4000, nothing)  # 3300 #1038
kernel = np.ones((5, 5), np.uint8)
while True:
    green_coefficient = cv2.getTrackbarPos("Green_coefficient", "Eval image") / 1000
    blue_coefficient = cv2.getTrackbarPos("Blue_coefficient", "Eval image") / 1000
    eval_array = red_channel - green_coefficient*green_channel - blue_coefficient*blue_channel
    eval_array_vis = (255*(eval_array - eval_array.min())/(eval_array.max() - eval_array.min())).astype(np.uint8)
    eval_array_vis = cv2.morphologyEx(eval_array_vis, cv2.MORPH_CLOSE, kernel)
    vis = np.concatenate((red_channel, green_channel, blue_channel), axis=1)
    # cv2.imshow("Original", img)
    # cv2.imshow("Channel split", vis)
    distance_1, img_out_1, error_1 = CircleBasedObjectDistance_1.calculate(eval_array_vis, (0, 0), (400, 600),
                                                                           0.2, "ball", 1)
    # cv2.imshow("Eval image", eval_array_vis)
    print(distance_1)
    cv2.imshow("Result", img_out_1)
    k = cv2.waitKey(100)
    if k == 27:
        break

