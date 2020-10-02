import cv2

while True:
    img = cv2.imread("20.jpg")
    img_crop = img[0:400, 0:400, :]
    cv2.imshow("Window_1", img_crop)
    cv2.waitKey(100)
