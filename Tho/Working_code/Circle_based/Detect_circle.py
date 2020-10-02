import numpy as np
import argparse
import cv2
import time


def nothing():
    pass


# Example usage: python3 Detect_circle.py --image ./Changed_data/30.jpg
# For Pycharm: Edit Configuration -> Parameters -> --image ./Changed_data/30.jpg
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())
img = cv2.imread(args["image"])
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.namedWindow('Window 1')
cv2.createTrackbar('Var_1', 'Window 1', 468, 1000, nothing)
cv2.createTrackbar('Var_2', 'Window 1', 22, 100, nothing)
while True:
    output = img.copy()
    Var_1 = cv2.getTrackbarPos('Var_1', 'Window 1')
    Var_2 = cv2.getTrackbarPos('Var_2', 'Window 1')
    start_time = time.time()
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10000,
                               param1=Var_1, param2=Var_2, minRadius=0, maxRadius=0)
    stop_time = time.time()
    print("{}".format(stop_time-start_time))
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        cv2.imshow("Window 1", np.hstack([img, output]))
    else:
        cv2.imshow("Window 1", np.hstack([img, img]))
    k = cv2.waitKey(1000) & 0xFF
    if k == 27:
        break
