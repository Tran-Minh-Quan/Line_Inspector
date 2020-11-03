import cv2
from Tho.Main_lib.distance_calc_lib import DistanceMultiClasses


def nothing():
    pass


distance_calc = DistanceMultiClasses(764)
img = cv2.imread("Clamp/clamp_40_1.jpg")
cv2.namedWindow("Result")
cv2.createTrackbar("Canny_1", "Result", 391, 1000, nothing)
cv2.createTrackbar("Canny_2", "Result", 664, 2000, nothing)
while True:
    Canny_1 = cv2.getTrackbarPos("Canny_1", "Result")
    Canny_2 = cv2.getTrackbarPos("Canny_2", "Result")
    distance, img_out, _ = distance_calc.calculate(img, (100, 50), (400, 400), 0, "clamp", 1)
    cv2.imshow("Result", img_out)
    print(distance)
    k = cv2.waitKey(100)
    if k == 27:
        break