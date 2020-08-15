from Distance_functions import distance_estimate
import cv2
import sys

def nothing(x):
    pass


img = cv2.imread("2_clamp_1177.jpg")
cv2.namedWindow('Window 1')
cv2.namedWindow('Window 2')
cv2.createTrackbar('Var_1', 'Window 1', 300, 1000, nothing)
cv2.createTrackbar('Var_2', 'Window 1', 900, 1000, nothing)
#print(distance)
test = True
while 1:
    Min_val = cv2.getTrackbarPos('Var_1', 'Window 1')
    Max_val = cv2.getTrackbarPos('Var_2', 'Window 1')
    distance, img_1, error = distance_estimate(img.copy(), [0, 0], [0, 0], 0.2, Min_val, Max_val, 14.7)
    if error == 0:
        img_1 = cv2.putText(img_1.copy(), "{} (mm)".format(distance), (105, 230), cv2.FONT_HERSHEY_SIMPLEX ,
                            0.5, (0, 0, 0) , 1, cv2.LINE_AA)
        cv2.imshow('Window 1', img_1)
        cv2.imshow('Window 2', img)
        #print("Distance: {} mm".format(distance))

        k = cv2.waitKey(100) & 0xFF
        if k == 27:
            break
    else:
    # print(distance)
        k = cv2.waitKey(100) & 0xFF
        if k == 27:
            break
