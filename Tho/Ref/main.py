from function import Find_crop_coordinate
import numpy as np
import cv2
import os

class Object:
    def __int__(self):
        self.width = 0
        self.height = 0


class Camera:
    def __int__(self):
        self.focal_length = 0
        self.image_height = 0
        self.image_width = 0


image_name = '0_ball_1'
my_path = os.path.abspath(os.path.dirname(__file__))
image_file = os.path.join(my_path, "Real_data", "")+image_name+'.jpg'
bounding_box_file = os.path.join(my_path, "Real_data", "")+image_name+'.txt'
print(image_file)

def nothing(x):
    pass


img = cv2.imread(image_file)
Camera.image_height = img.shape[0]
Camera.image_width = img.shape[1]

Camera.focal_length = 764
Object.width = 7
cv2.namedWindow('Window 1')

cv2.createTrackbar('Focal_length(px)', 'Window 1', 764, 1000, nothing)
cv2.createTrackbar('Object_width(cm)', 'Window 1', 7, 100, nothing)

while(1):
    img = cv2.imread(image_file)
    [start_point, end_point, distance] = Find_crop_coordinate(bounding_box_file, Object.width,
                                                              Camera.focal_length, Camera.image_width,
                                                              Camera.image_height)
    img = cv2.rectangle(img, tuple(start_point), tuple(end_point), (255, 0, 0), 2)
#    cv2.putText(img, 'Distance: '+str(distance)+'(cm)', (Camera.image_width-240, Camera.image_height-10), font, 0.75,
#                (255, 255, 255), 2)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    Camera.focal_length = cv2.getTrackbarPos('Focal_length(px)', 'Window 1')
    Object.width = cv2.getTrackbarPos('Object_width(cm)', 'Window 1')
    resize_width = int(540*Camera.image_width/Camera.image_height)
    img_resized = cv2.resize(img, (resize_width, 580))
    img_resized = cv2.putText(img_resized, 'Distance: ' + str(distance) + '(mm)', (resize_width - 240, 580 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    img_resized = cv2.imshow('Window 1', img_resized)
cv2.destroyAllWindows()





