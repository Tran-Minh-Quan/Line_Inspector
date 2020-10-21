import cv2

img = cv2.imread("20.jpg", 0)
img_out = cv2.Laplacian(img, cv2.CV_8U, ksize=5)
cv2.imshow("Window 1", img_out)
cv2.waitKey(0)
