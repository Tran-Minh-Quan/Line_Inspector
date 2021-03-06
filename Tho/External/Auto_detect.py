import numpy as np
import argparse
import cv2
import time
import os

def nothing():
    pass


# Example usage: python3 Auto_detect.py --image ./Changed_data/20.jpg
# For Pycharm: Edit Configuration -> Parameters -> --image ./Changed_data/20.jpg
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True, help="Path to image")
#args = vars(ap.parse_args())
# Những chỗ có # ở sau là đã thay đổi so với code gốc
for img_file in os.listdir('./Changed_data'):          #
    img_dir = os.path.join('./Changed_data',img_file)  #
    img = cv2.imread(img_dir)                          #
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.namedWindow(img_file)                          #
    Canny_low = 300
    Canny_high = 600
    Canny_step = 5
    Canny_param = Canny_high
    start_time = time.time()
    while True:
        output = img.copy()
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=Canny_param, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            circles_round = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles_round:
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            cv2.imshow(img_file, np.hstack([img, output]))     #
            if circles.size is not 3:
                if Canny_param < Canny_high:
                    Canny_param += Canny_step
            else:
                stop_time = time.time()
                print("{}".format(stop_time - start_time))
                print("R = {} pixel".format(circles[0][0][2]))
                break
        else:
            if Canny_param >= Canny_low + Canny_step:
                Canny_param -= Canny_step
            else:
                pass
            cv2.imshow(img_file, np.hstack([img, img]))  #
        print(Canny_param)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()                      #
            break
