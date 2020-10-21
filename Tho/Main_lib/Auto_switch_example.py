import cv2
from distance_calc_lib import CircleBasedObjectDistance
import os
from Quan import QuanLib


CircleBasedObjectDistance_1 = CircleBasedObjectDistance(764)
for img_file in os.listdir('./Data/Damper_2'):
    img_dir = os.path.join('./Data/Damper_2', img_file)
    img = QuanLib.gamma_correct(img_dir=img_dir, gamma=3.5, save_dir="./")

    while True:
        distance_1, img_out_1, error_1 = CircleBasedObjectDistance_1.calculate(img, (0, 0), (480, 640),
                                                                               0.2, "damper", 1)
        print("Distance: %.2f\nError code: %d\n" % (distance_1, error_1))
        cv2.imshow("Window 1", img_out_1)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
