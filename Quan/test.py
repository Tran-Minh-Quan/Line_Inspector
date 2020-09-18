import os
import cv2
from Quan import QuanLib
import sys
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
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

def test_model():
  cap = cv2.VideoCapture(0)
  if cap.isOpened() == False:
      sys.exit("nani")
  model = QuanLib.Yolov3(weights_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov4.weights',
                          cfg_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov4.cfg',
                            names_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\coco.names')
  pre_time = time.time()
  while True:
    ok, frame = cap.read()
    if not ok:
      sys.exit("Cannot grab the frame or reached the end of the video")
    model.detect(frame)
    cv2.imshow('frame',frame)
    cycle_time = time.time() - pre_time
    pre_time = time.time()
    if cycle_time != 0:
      FPS = round(1 / cycle_time)
    print('T = '+ str(cycle_time)+'ms')
    print('FPS = '+str(FPS))

    cv2.waitKey(1)

def test_contour():
  def Trackchanged(x):
    pass
  org_img = cv2.imread('D:\\WON\\DO_AN\\Changed_data\\20.jpg')
  #org_img = cv2.imread('D:\\WON\\DO_AN\Data\\20.jpg')

  winName = 'Test'
  cv2.namedWindow(winName,2)
  cv2.createTrackbar('tlx', winName, 162, org_img.shape[1], Trackchanged)
  cv2.createTrackbar('tly', winName, 99, org_img.shape[0], Trackchanged)
  cv2.createTrackbar('brx', winName, 374, org_img.shape[1], Trackchanged)
  cv2.createTrackbar('bry', winName, 400, org_img.shape[0], Trackchanged)
  cv2.createTrackbar('minVal', winName, 100, 600, Trackchanged)
  cv2.createTrackbar('maxVal', winName, 200, 600, Trackchanged)
  cv2.createTrackbar('blur', winName, 3, 50, Trackchanged)

  while True:
    top_left_x = cv2.getTrackbarPos('tlx', winName)
    top_left_y = cv2.getTrackbarPos('tly', winName)
    bottom_right_x = cv2.getTrackbarPos('brx', winName)
    bottom_right_y = cv2.getTrackbarPos('bry', winName)
    minVal = cv2.getTrackbarPos('minVal',winName)
    maxVal = cv2.getTrackbarPos('maxVal',winName)
    blur_kernel_size = cv2.getTrackbarPos('blur',winName)

    img = org_img.copy()
    gray_img = cv2.cvtColor(img[top_left_y:bottom_right_y, top_left_x:bottom_right_x], cv2.COLOR_BGR2GRAY)
    try:
      blur = cv2.GaussianBlur(gray_img, (blur_kernel_size, blur_kernel_size), 0)
    except:
      pass
    edges = cv2.Canny(blur, minVal, maxVal)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), iterations=1)
    edges = cv2.erode(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), iterations=1)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(edges_rgb, contours, -1, (0, 255, 0), 3)
    img[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = edges_rgb

    cv2.imshow('image',img)
    cv2.imshow('gray image',gray_img)
    #cv2.imshow('After blurring image',blur)
    #cv2.imshow('edges rgb',edges_rgb)

    if cv2.waitKey(1) == 27:
      break
def testthreshold(winName='tên của window',cam=1):
  capimg = cv2.VideoCapture(cam)
  if (capimg.isOpened() == False):
    sys.exit("Unable to read camera feed")
  if isinstance(cam, str):
    frameNum = int(capimg.get(cv2.CAP_PROP_FRAME_COUNT))
  def Trackchanged(x):
    pass
  cv2.namedWindow(winName)
  if isinstance(cam, str):
    cv2.createTrackbar('frame', winName, 8, frameNum, Trackchanged)
  cv2.createTrackbar('blur',winName,3,51,Trackchanged)
  cv2.createTrackbar('minVal', winName, 0, 600, Trackchanged)
  cv2.createTrackbar('maxVal', winName, 40, 600, Trackchanged)
  invGamma = 1.0 / 1.5
  # tao lookup table
  table = np.array([((i / 255.0) ** invGamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")


  while (True):
    blur_kernel_size = cv2.getTrackbarPos('blur',winName)
    minVal = cv2.getTrackbarPos('minVal', winName)
    maxVal = cv2.getTrackbarPos('maxVal', winName)
    if isinstance(cam, str):
      frameIndex = cv2.getTrackbarPos('frame', winName)
      capimg.set(cv2.CAP_PROP_POS_FRAMES, frameIndex-1)
    ret, frame = capimg.read()
    if ret == True:
      # apply gamma correction using the lookup table
      frame = cv2.LUT(frame, table)
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      roi = gray_frame[75:95, 220:420]
      template = cv2.imread('D:\\WON\\DO_AN\\Code\\Huy\\template.jpg', 0)
      w, h = template.shape[::-1]
      res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      top_left = (max_loc[0] + 220, max_loc[1] + 75)
      bottom_right = (top_left[0] + w, top_left[1] + h)

      line = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()
      try:
        blur = cv2.GaussianBlur(line, (blur_kernel_size, blur_kernel_size), 0)
      except:
        pass
      edges = cv2.Canny(blur, minVal, maxVal)
      kernel = np.array([[0,0,1],[0,1,0],[1,0,0]],dtype=np.uint8)
      edges = cv2.dilate(edges, kernel, iterations=1)
      canny = cv2.erode(edges, kernel, iterations=1)
      #contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
      #cv2.drawContours(canny_rgb, contours, -1, (0,255,0), 1)
      #edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
      result = cv2.hconcat([line,canny_rgb])


      #cv2.imshow('resized_line',resized_line)
      cv2.imshow('line', result)
      cv2.waitKey(1)
      #edges = cv2.Canny(opening, 50, 200, apertureSize=3)
      #feat = np.sum(edges / 255)
      #print(feat)

      # Press Q on keyboard to stop recording
      #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

    # Break the loop
    else:
      break

  # When everything done, release the video capture and video write objects
  capimg.release()
  # Closes all the frames
  cv2.destroyAllWindows()
  #video2.avi: blocksize = 897, c = 27, canny dùng opening 2
  #checkline.avi: blocksize = 897, c = 53, canny dùng opening 1





#continuous_check(check_folder_path = 'D:/WON/DO_AN/Code/Code_Sa/test/Label/')
#Sa.folder_rename(rename_folder_path = 'D:/WON/DO_AN/Training/data/insulator/Label/',extension = '.txt',new_name = '3_insulator_',begin_index = 1)
#combine_to_one_screen('VIDEO1_svm.avi','VIDEO1_yolo.avi')
#testthreshold(winName='Test trackbar',cam='D:\\WON\\DO_AN\\Code\\Huy\\video_test_day.avi')

