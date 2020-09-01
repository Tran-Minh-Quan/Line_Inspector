import cv2
import numpy as np
from matplotlib import pyplot as plt
def tachday(cam) :
 capimg = cv2.VideoCapture(cam)
 if (capimg.isOpened() == False) :
   print("Unable to read camera feed")
 if isinstance(cam, str):
    frameNum = int(capimg.get(cv2.CAP_PROP_FRAME_COUNT))
 def Trackchanged(x):
   pass
 cv2.namedWindow('Tracking')
 cv2.createTrackbar('frame', "Tracking", 8, frameNum, Trackchanged)
 if isinstance(cam, str):
   frameIndex = cv2.getTrackbarPos('frame', "Tracking")
   capimg.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
 while(True):
     ret, img = capimg.read()
     if ret == True:
         frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         frame1 = frame[75:95, 220:420].copy()
         kernel = np.ones((3, 3), np.uint8)
         template = cv2.imread('template.jpg',0)
         w , h = template.shape[::-1]
        # All the 6 methods for comparison in a list
         res = cv2.matchTemplate(frame1, template, cv2.TM_CCOEFF )
         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
         top_left = max_loc
         bottom_right = (top_left[0] + w-10, top_left[1] + h)
         cv2.rectangle(frame1, top_left, bottom_right, (0.0,255) , 2)
         plt.subplot(121), plt.imshow(res, cmap='gray')
         plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
         plt.subplot(122), plt.imshow(img, cmap='gray')
         plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
         cv2.rectangle(frame, (220, 75), (420, 95), (0, 0, 255), 2)
         cv2.imshow('frame',frame)
         1
         print(top_left)
         cv2.waitKey(20)

# k = cv2.waitKey(1) & 0xff
 #if k == 27:
  #  break
 capimg.release()

  # Closes all the frames
 cv2.destroyAllWindows()
tachday(cam='video_test_day.avi')






