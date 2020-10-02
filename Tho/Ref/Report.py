import cv2
import numpy as np

img = cv2.imread("2_clamp_1177.jpg", 0)
crop_img = img[0:250, 210:425]
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(crop_img.copy(), kernel, iterations=1)
opening_image = cv2.dilate(erosion.copy(), kernel, iterations=1)
cv2.namedWindow('Window 1')
cv2.namedWindow('Window 2')
cv2.imshow('Window 1', crop_img)
cv2.imshow('Window 2', opening_image)
#cv2.imshow('Window 2', dilation)
k = cv2.waitKey(0)
cv2.destroyAllWindows()