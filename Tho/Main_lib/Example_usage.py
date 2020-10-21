import cv2
from distance_calc_lib import CircleBasedObjectDistance
import time


# Canny_low = 200, Canny_high = 5000, step_size = 1, hough_param =22,
# slope = 867.7887424687945, intercept = -0.18242145320198588
# Circle_based_estimate_1 = CircleDistance(0, 50000, 1, 50, 867.7887424687945, -0.18242145320198588)
time_start = time.time()

CircleBasedObjectDistance_1 = CircleBasedObjectDistance(764)
img = cv2.imread("./Data/45.jpg")
distance_1, img_out_1, error_1 = CircleBasedObjectDistance_1.calculate(img, (50, 50), (400, 400),
                                                                       0.2, "damper", 1)
time_stop_1 = time.time()
img = cv2.imread("./Data/57.jpg")
distance_2, img_out_2, error_2 = CircleBasedObjectDistance_1.calculate(img, (100, 200), (300, 400),
                                                                       0.2, "damper", 1)
time_stop_2 = time.time()
img = cv2.imread("./Data/59.jpg")
distance_3, img_out_3, error_3 = CircleBasedObjectDistance_1.calculate(img, (100, 200), (300, 400),
                                                                       0.2, "damper", 1)
# time_stop_2 = time.time()
# img = cv2.imread("./Changed_data/test_2.jpg")
# Circle_based_estimate_1.first_detect = 1
# distance_3, img_out_3, error_3 = Circle_based_estimate_1.calculate(img, (50, 300), (210, 550),
#                                                              0.1, mode=1, object_width=10)
time_stop_3 = time.time()

cv2.imshow("Window 1", img_out_1)
cv2.waitKey(100)
cv2.imshow("Window 1", img_out_2)
cv2.waitKey(100)
cv2.imshow("Window 1", img_out_3)
cv2.waitKey(100)
print("Distance: %.2f\nError code: %d\nTime run: %.4f" % (distance_1, error_1, time_stop_1-time_start))
print("Distance: %.2f\nError code: %d\nTime run: %.4f" % (distance_2, error_2, time_stop_2-time_stop_1))
print("Distance: %.2f\nError code: %d\nTime run: %.4f" % (distance_3, error_3, time_stop_3-time_stop_2))
