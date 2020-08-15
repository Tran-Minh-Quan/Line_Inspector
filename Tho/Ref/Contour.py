import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from function import Find_crop_coordinate

class Canny:
    def __init__(self):
        self.var_1 = 0
        self.var_2 = 0


def nothing(x):
    pass


kernel = np.ones((5, 5), np.uint8)


image_name = '2_clamp_116'
my_path = os.path.abspath(os.path.dirname(__file__))
image_file = os.path.join(my_path, "Real_data", "")+image_name+'.jpg'
bounding_box_file = os.path.join(my_path, "Real_data", "")+image_name+'.txt'
print(image_file)

[start_point, end_point, distance] = Find_crop_coordinate(bounding_box_file, 7,
                                                          764, 640,
                                                          480)
img = cv2.imread(image_file, 0)
width_box = end_point[1] - start_point[1]
height_box = end_point[0] - start_point[0]
x_axis_extended = [max(0, int(start_point[1]-0.2*width_box)), min(img.shape[0], int(end_point[1] + 0.2*width_box))]
y_axis_enxtended = [max(0, int(start_point[0]-0.2*height_box)), min(img.shape[1], int(end_point[0] + 0.2*height_box))]
crop_img_1 = img[x_axis_extended[0]: x_axis_extended[1], y_axis_enxtended[0]:y_axis_enxtended[1]]
crop_img = cv2.resize(crop_img_1, (500, 500))
cv2.namedWindow('Window 1')
cv2.namedWindow('Window 2')
cv2.createTrackbar('Var_1', 'Window 1', 60, 1000, nothing)
cv2.createTrackbar('Var_2', 'Window 1', 120, 1000, nothing)
i = 1
while(1):
    img = cv2.imread(image_file, 0)
    crop_img = img[x_axis_extended[0]: x_axis_extended[1], y_axis_enxtended[0]:y_axis_enxtended[1]]
    Canny.var_1 = cv2.getTrackbarPos('Var_1', 'Window 1')
    Canny.var_2 = cv2.getTrackbarPos('Var_2', 'Window 1')
    erode = cv2.erode(crop_img, kernel, iterations=1)
    edges_1 = cv2.Canny(erode, Canny.var_1, Canny.var_2)
    
    contours, hierarchy = cv2.findContours(edges_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    """
    for x in range (1, len(contours)):
    img_1 = cv2.drawContours(img, contours[x], -1, (0, 255, 0), 3)
    """
    img_1 = cv2.drawContours(crop_img.copy(), contours, -1, (0, 255, 0), 3)
    c = max(contours, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    print(extLeft)
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break
    cv2.imshow('Window 1', img_1)
    cv2.imshow('Window 2', erode)
"""
Canny.var_1 = cv2.getTrackbarPos('Var_1', 'Window 1')
Canny.var_2 = cv2.getTrackbarPos('Var_2', 'Window 1')
img = cv2.imread(image_file, 0)
edges = cv2.Canny(img, Canny.var_1, Canny.var_2)
image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_1 = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
print(str(len(contours)))
"""