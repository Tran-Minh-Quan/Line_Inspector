import os
import cv2
from Quan import QuanLib
import sys
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import anhSang.Sahelpers as Sa
import cupy as cp
import albumentations as A
import albumentations.augmentations.transforms as transforms

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

def test_augmentation():
  while True:
    #origin_foreground = cv2.imread(r'D:\WON\DO_AN\Data\Training\Augmented_image\Insulator\3_insulator_10-won-removedbg.png')
    #origin_image = cv2.imread(r'D:\WON\DO_AN\Data\Training\insulator\Image\3_insulator_10.jpg')
    #gray_foreground = cv2.cvtColor(origin_foreground,cv2.COLOR_BGR2GRAY)
    #ret,mask = cv2.threshold(gray_foreground,0,255,cv2.THRESH_BINARY)
    #mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    #good_foreground = cv2.bitwise_and(origin_image,mask)
    #cv2.imwrite(r'D:\WON\DO_AN\Data\Training\Augmented_image\Insulator\3_insulator_10-won-removedbg.png',good_foreground)
    fg = cv2.imread(r'D:\WON\DO_AN\Data\Training\Lan1\Ball\Segmented\ball_88_1_segmented.jpg')
    height, width, channels = fg.shape
    gray_fg = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    _,mask = cv2.threshold(gray_fg, 254, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

    rand_bg = np.random.rand(height,width,channels)*256
    rand_bg = rand_bg.astype(np.uint8)
    combine_img = cv2.add(cv2.bitwise_and(mask,rand_bg),fg)
    cv2.imshow('background',combine_img)
    cv2.waitKey()

def test_brightness_hsv(img_dir,gamma):
  img = cv2.imread(img_dir)
  img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  h,s,v = cv2.split(img_hsv)
  invGamma = 1 / gamma
  table = np.array([((i / 255.0) ** invGamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")
  v_corrected = cv2.LUT(v, table)
  s_corrected = cv2.LUT(s, table)
  img_gammaCorrected = cv2.LUT(img.copy(), table)
  img_hsv = cv2.merge([h,s_corrected,v_corrected])
  img_corrected = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
  cv2.imshow('v correct',cv2.hconcat([img,img_corrected]))
  cv2.imshow('gamma correct',cv2.hconcat([img,img_gammaCorrected]))
  cv2.waitKey()
# test_brightness_hsv(r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Images\clamp_96_2.jpg',4)
# continuous_check(check_folder_path = 'D:/WON/DO_AN/Code/Code_Sa/test/Label/')
# Sa.folder_rename(rename_folder_path = 'D:/WON/DO_AN/Training/data/insulator/Label/',extension = '.txt',new_name = '3_insulator_',begin_index = 1)
# combine_to_one_screen('VIDEO1_svm.avi','VIDEO1_yolo.avi')
# testthreshold(winName='Test trackbar',cam='D:\\WON\\DO_AN\\Code\\Huy\\video_test_day.avi')
# QuanLib.gamma_correct(data_dir=r'D:\WON\DO_AN\Data\Distance\Clamp\No_repeat_images',gamma=2.5,save_dir=r'D:\WON\DO_AN\Data\Distance\Clamp\No_repeat_changed_images')

# Lọc ra ảnh thứ nhất
# for i in os.listdir(r'D:\WON\DO_AN\Data\Distance\Clamp\Images'):
#   if i.split('.')[0].split('_')[2] == '1':
#     print(i)
#     i_png = i.replace('.jpg', '.png')
#     cv2.imwrite(r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Images\Raw\\' + i_png,
#                 cv2.imread(r'D:\WON\DO_AN\Data\Distance\Clamp\Images\\' + i))

# # Test sai số khoảng cách nếu người đánh box
# mpl.rcParams["savefig.directory"] = "D:\WON\DO_AN\Data\Distance\Result"
# inv_w_array = np.array([])
# dis_array = np.array([])
# for txt_file in os.listdir('D:\WON\DO_AN\Data\Distance\Clamp\Labels'):
#   txt_content = open('D:\WON\DO_AN\Data\Distance\Clamp\Labels\\'+txt_file)
#   _,x_center,y_center,bb_w,bb_h = txt_content.read().splitlines()[0].split(r' ')
#   dis = txt_file.split('.')[0].split('_')[1]
#   w = 640*float(bb_w)
#   dis_array = np.append(dis_array,int(dis))
#   inv_w_array = np.append(inv_w_array,1/w)
#   txt_content.close()
# zipped_lists = zip(dis_array, inv_w_array)
# sorted_pairs = sorted(zipped_lists)
# tuples = zip(*sorted_pairs)
# dis_array, inv_w_array = [np.asarray(tuple) for tuple in tuples]
# np.savetxt('D:\WON\DO_AN\Data\Distance\Clamp\inv_w_array.csv',inv_w_array)
# np.savetxt('D:\WON\DO_AN\Data\Distance\Clamp\dis_array.csv',dis_array)
# QuanLib.linear_regression(inv_w_array_dir='D:\WON\DO_AN\Data\Distance\Clamp\inv_w_array.csv',
#                    dis_array_dir='D:\WON\DO_AN\Data\Distance\Clamp\dis_array.csv')

# img = cv2.imread(r'D:\WON\DO_AN\Data\Training\Lan1\Damper\Images\Foreground\Raw\damper_20_1_segmented.jpg')
# org = cv2.imread(r'D:\WON\DO_AN\Data\Training\Lan1\Damper\Images\gamma_corrected_no_repeat\damper_20_1.jpg')
# cv2.imshow('org',org)
# mask = cv2.inRange(img,(243,150,33),(243,150,33))
# mask = cv2.resize(mask,(640,480))
# mask = cv2.bitwise_not(mask)
# mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
# # cv2.imwrite(r'D:\WON\DO_AN\Data\Training\Lan1\Damper\Images\Mask\damper_20_1_mask.jpg',mask)
# cv2.imshow('mask',mask)
# fg = cv2.bitwise_and(mask,org)
# # cv2.imwrite(r'D:\WON\DO_AN\Data\Training\Lan1\Damper\Images\Foreground\Corrected\damper_20_1_foreground.png',fg)
# cv2.imshow('fg',fg)
# fg = cv2.imread(r'D:\WON\DO_AN\Data\Training\Lan1\Damper\Images\Foreground\Corrected\damper_20_1_foreground.jpg')
# dif = org-fg
# cv2.imshow('dif',dif)
# cv2.waitKey()

# transform = A.Compose([
#       A.RandomCrop(width=256, height=256,p=0.1),
#       A.HorizontalFlip(p=0.5),
#       A.RandomBrightnessContrast(p=0.2),
#   ])
# image = cv2.imread(r"D:\WON\DO_AN\Data\Training\Lan1\Ball\Raw\Images\ball_20_1.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# while True:
#   transformed = transform(image=image)
#   transformed_image = transformed["image"]
#   transformed_image = cv2.cvtColor(transformed_image,cv2.COLOR_RGB2BGR)
#   cv2.imshow('sdfsf',transformed_image)
#   cv2.waitKey()

# org = cv2.imread(r'D:\WON\DO_AN\Data\Training\Lan1\Damper\Images\gamma_corrected_no_repeat\damper_20_1.jpg')
# mask = cv2.imread('D:\WON\DO_AN\Data\Training\Lan1\Damper\Images\Mask\damper_20_1_mask.png')
# fg = cv2.bitwise_and(org,mask)
# cv2.imshow('dif',org-fg)
# cv2.waitKey

# Test trừ hai ảnh
# cap = cv2.VideoCapture(r'D:\WON\DO_AN\Data\Line\line_2.avi')
# frame1 = cap.read()[1]
# frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# while 1:
#   for i in range(15):
#     frame2 = cap.read()[1]
#   frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
#   r = cv2.selectROI('roi',frame2,showCrosshair=0)
#   print(r)
#   frame_dif = cv2.subtract(frame2,frame1)
#   frame1 = frame2
#   cv2.imshow('frame dif',frame_dif)
#   cv2.imshow('frame',frame2)
#   cv2.waitKey()

# Test clahe
# winName = 'clahe test'
# cv2.namedWindow(winName)
#
# def callback(x):
#     pass
#
# cv2.createTrackbar('clipLimit', winName, 40 ,100, callback)
# cv2.createTrackbar('gridSize', winName, 1, 100, callback)
# cv2.createTrackbar('gamma', winName, 500, 1000, callback)
# while True:
#     clip_limit = cv2.getTrackbarPos('clipLimit', winName)
#     grid_size = cv2.getTrackbarPos('gridSize', winName)
#     gamma = cv2.getTrackbarPos('gamma', winName)
#     img = cv2.imread(r'D:\WON\DO_AN\Data\Distance\Damper_2\Images\raw\damper_20_1.jpg')
#     if grid_size:
#         clip_limit = 16
#         grid_size = 2
        # clahe_mat = cv2.createCLAHE(clip_limit, 2 * (grid_size,))
    # else:
    #     cv2.waitKey(1)
    #     continue
    # if gamma:
    #     invGamma = 100 / gamma  # Đúng ra là 1/gamma nhưng làm thế để chạy được số thập phân (có thể làm tối)
    # else:
    #     cv2.waitKey(1)
    #     continue
    # table = np.array([((i / 255.0) ** invGamma) * 255
    #                   for i in np.arange(0, 256)]).astype("uint8")
    # img = cv2.LUT(img, table)
    # cv2.imshow('after gamma correct', img)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img_hsv[:, :, 2] = clahe_mat.apply(img_hsv[:, :, 2])
    # img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    # img = cv2.GaussianBlur(img, (5,5), 0)
    # cv2.imshow(winName, img)
    # cv2.imwrite(r'D:\WON\DO_AN\Data\Training\Lan1\Damper\Images\gamma_corrected_no_repeat\damper_100_1_clahe.jpg', img)
    #
    # cv2.waitKey(1)

# Test tách màu ball
img_folder_dir = r'D:\WON\DO_AN\Data\Training\Lan1\Ball\Raw\Images'
img_name_list = [i for i in os.listdir(img_folder_dir)]
winName = 'testSegmentColor'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
cv2.resizeWindow(winName, 500, 300)
def callback(x):
    pass
cv2.createTrackbar('lowHue', winName, 0, 179, callback)
cv2.createTrackbar('lowSat', winName, 4, 255, callback)
cv2.createTrackbar('lowVal', winName, 10, 255, callback)
cv2.createTrackbar('highHue', winName, 19, 255, callback)
cv2.createTrackbar('highSat', winName, 255, 255, callback)
cv2.createTrackbar('highVal', winName, 255, 255, callback)
cv2.createTrackbar('maxVal', winName, 255, 500, callback)
cv2.createTrackbar('minVal', winName, 10, 500, callback)
index = 0
invGamma = 1/2
table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
while True:
    img_name = img_name_list[index]
    img_dir = os.path.join(img_folder_dir, img_name)
    img = cv2.imread(img_dir)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img = cv2.LUT(img, table)

    lowHue = cv2.getTrackbarPos('lowHue', winName)
    lowSat = cv2.getTrackbarPos('lowSat', winName)
    lowVal = cv2.getTrackbarPos('lowVal', winName)
    highHue = cv2.getTrackbarPos('highHue', winName)
    highSat = cv2.getTrackbarPos('highSat', winName)
    highVal = cv2.getTrackbarPos('highVal', winName)
    maxVal = cv2.getTrackbarPos('maxVal', winName)
    minVal = cv2.getTrackbarPos('minVal', winName)

    # Range for lower red
    lower_red = np.array([0, 4, 10])
    upper_red = np.array([10, 255, 255])
    # mask1 = cv2.inRange(img_hsv, (lowHue, lowSat, lowVal), (highHue, highSat, highVal))
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # Range for upper range
    lower_red = np.array([135, 15, 40])
    upper_red = np.array([255, 255, 255])
    # mask2 = cv2.inRange(img_hsv, (lowHue, lowSat, lowVal), (highHue, highSat, highVal))
    mask2 = cv2.inRange(img_hsv, lower_red, upper_red)

    # Generating the final mask to detect red color
    mask = mask1 + mask2
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
    edges = cv2.Canny(mask, maxVal, minVal)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    segmented = cv2.bitwise_and(img, mask)

    info_panel = np.zeros((98, 640 * 2, 3), np.uint8)
    info_panel[:, :] = [50, 50, 50]
    info = 'Current mask: ' + img_name_list[index] + ' -- Index: ' + str(index)
    cv2.putText(info_panel, info, (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 1, lineType=cv2.LINE_AA)
    check_img = cv2.vconcat([info_panel, cv2.hconcat([edges, mask])])
    cv2.imshow('check', check_img)
    toggle = cv2.waitKeyEx(1)
    if toggle == 2555904:
      index += 1
      if index == len(img_name_list):
        index = 0
    elif toggle == 2424832:
      index -= 1
      if index < 0:
        index = len(img_name_list) - 1

