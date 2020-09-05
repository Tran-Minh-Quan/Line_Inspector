import os
import cv2
import anhSang.Sahelpers as Sa
def continuous_check(check_folder_path = 'D:/WON/DO_AN/Code/Code_Sa/test/ball/Image/'):
  filelist = []
  number_string = ''
  folder_list = os.listdir(check_folder_path)
  for filename in folder_list:
    for character in filename:
      if  character.isnumeric() == True:
        number_string = number_string + character
    filelist.append(number_string)
    number_string = ''
  filelist = sorted(filelist,key=int)
  print(filelist)
  print(len(filelist))
  for i in range(len(filelist)-2):
    if int(filelist[i+1])-int(filelist[i]) != 1:
      print(filelist[i])
def combine_to_one_screen(vid1='',vid2=''):
  cap1 = cv2.VideoCapture(vid1)
  cap2 = cv2.VideoCapture(vid2)
  if not cap1.isOpened() or not cap2.isOpened():
    print('Could not open camera')
    return
  saved_video = cv2.VideoWriter('compare_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 480))
  while 1:
    ok1, frame1 = cap1.read()
    ok2, frame2 = cap2.read()
    if not ok1 or not ok2:
      break
    screen = cv2.hconcat([frame1[:,0:640],frame2[:,0:640]])
    cv2.imshow('screen',screen)
    cv2.waitKey(50)
    saved_video.write(screen)
  cap1.release()
  cap2.release()
  cv2.destroyAllWindows()

#continuous_check(check_folder_path = 'D:/WON/DO_AN/Code/Code_Sa/test/Label/')
#Sa.folder_rename(rename_folder_path = 'D:/WON/DO_AN/Training/data/insulator/Label/',extension = '.txt',new_name = '3_insulator_',begin_index = 1)
#combine_to_one_screen('VIDEO1_svm.avi','VIDEO1_yolo.avi')

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
 while (True):
  if isinstance(cam, str):
   frameIndex = cv2.getTrackbarPos('frame', "Tracking")
   capimg.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
   ret, img = capimg.read()
   if ret == True:
     frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     frame1 = frame[75:95, 220:420].copy()
     kernel = np.ones((3, 3), np.uint8)
     template = cv2.imread('doanday1.jpg',0)
     w , h = template.shape[::-1]

     res = cv2.matchTemplate(frame1, template, cv2.TM_CCOEFF )
     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
     top_left = max_loc
     bottom_right = (top_left[0] + w, top_left[1] + h)
     cv2.rectangle(frame1, top_left, bottom_right, (0.255,255) , 2)
     plt.subplot(121), plt.imshow(res, cmap='gray')
     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
     plt.subplot(122), plt.imshow(img, cmap='gray')
     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
     cv2.rectangle(frame, (220, 75), (420, 95), (0, 0, 255), 2)
     cv2.imshow('frame',frame)
     cv2.imshow('template',template)
     cv2.imshow('detected',frame1)
     print(w)

     plt.show()


 capimg.release()

  # Closes all the frames
 cv2.destroyAllWindows()
tachday('video2.avi')
