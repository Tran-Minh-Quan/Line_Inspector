import imutils
import cv2
import os
import numpy as np
from function import find_crop_coordinate
from function import calculate_distance


class Canny:
    def __init__(self):
        self.var_1 = 0
        self.var_2 = 0


class object:
    def __init__(self):
        self.width = 7


class camera:
    def __init__(self):
        self.focal_length = 764


def nothing(x):
    pass


extended_ratio = 0.1

kernel = np.ones((5, 5), np.uint8)

image_name = '2_clamp_1177'
my_path = os.path.abspath(os.path.dirname(__file__))
image_file = os.path.join(my_path, "Real_data", "") + image_name + '.jpg'
bounding_box_file = os.path.join(my_path, "Real_data", "") + image_name + '.txt'
img = cv2.imread(image_file, 0)

[start_point, end_point] = find_crop_coordinate(bounding_box_file, 7, 764, 640, 480)

width_box = end_point[1] - start_point[1]
height_box = end_point[0] - start_point[0]
x_axis_extended = [max(0, int(start_point[1] - extended_ratio * width_box)),
                   min(img.shape[0], int(end_point[1] + extended_ratio * width_box))]
y_axis_extended = [max(0, int(start_point[0] - extended_ratio * height_box)),
                   min(img.shape[1], int(end_point[0] + extended_ratio * height_box))]
cv2.namedWindow('Window 1')
cv2.namedWindow('Window 2')
cv2.createTrackbar('Var_1', 'Window 1', 100, 1000, nothing)
cv2.createTrackbar('Var_2', 'Window 1', 200, 1000, nothing)
while (1):
    img = cv2.imread(image_file, 0)
    crop_img = img[x_axis_extended[0]: x_axis_extended[1], y_axis_extended[0]:y_axis_extended[1]]
    Canny.var_1 = cv2.getTrackbarPos('Var_1', 'Window 1')
    Canny.var_2 = cv2.getTrackbarPos('Var_2', 'Window 1')
    erode = cv2.erode(crop_img, kernel, iterations=1)
    dilation = cv2.dilate(erode.copy(), kernel, iterations=1)
    edges = cv2.Canny(dilation, Canny.var_1, Canny.var_2)
    contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    c = sorted(contours, key=cv2.contourArea)

    extLeft, extRight, extTop, extBot = 1e9, -1e9, 1e9, -1e9

    for cc in c:
        extLeft = min(cc[:, :, 0].min(), extLeft)
        extRight = max(cc[:, :, 0].max(), extRight)
        extTop = min(cc[:, :, 1].min(), extTop)
        extBot = max(cc[:, :, 1].max(), extBot)

    cv2.rectangle(crop_img, (extLeft, extTop), (extRight, extBot), (0, 255, 0))
    img_1 = cv2.drawContours(crop_img.copy(), contours, -1, (0, 255, 0), 3)
    distance = calculate_distance(7, 764, extRight - extLeft)
    print("{} {} ".format(start_point, end_point))
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break
    cv2.imshow('Window 1', img_1)
    cv2.imshow('Window 2', edges)
