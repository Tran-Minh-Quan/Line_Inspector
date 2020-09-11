import cv2
from distance_calc import CircleDistance
import time
import numpy as np

img = cv2.imread("./Changed_data/25.jpg")
# Canny_low = 200, Canny_high = 600, step_size = 5, hough_param =22,
# slope = 867.7887424687945, intercept = -0.18242145320198588
Circle_based_estimate_1 = CircleDistance(200, 600, 1, 22, 867.7887424687945, -0.18242145320198588)
time_start = time.time()
distance, img_out, error = Circle_based_estimate_1.calculate(img, (100, 200), (300, 400),
                                                             0.2, mode=1, object_width=10)
time_stop = time.time()
print("Distance: %.2f" % distance)
print("Error code: %.2f" % error)
print(time_stop-time_start)
cv2.imshow("Window 1", img_out)
cv2.waitKey(1000)
