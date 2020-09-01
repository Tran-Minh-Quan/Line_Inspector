import cv2
import numpy as np
import func as f
import matplotlib.pyplot as plt
plt.ion()
def testthreshold(winName='tên của window',cam=1):
  capimg = cv2.VideoCapture(cam)
  if (capimg.isOpened() == False):
    print("Unable to read camera feed")
  if isinstance(cam, str):
    frameNum = int(capimg.get(cv2.CAP_PROP_FRAME_COUNT))
  x = np.array([np.arange(frameNum+1)])
  y = np.zeros((1, frameNum+1), np.uint8)
  def Trackchanged(x):
    pass
  cv2.namedWindow(winName)
  cv2.createTrackbar("THKersize", winName, 11, 897, Trackchanged)
  cv2.createTrackbar("C", winName, 48,255 , Trackchanged)
  cv2.createTrackbar('OpKerSize',winName,3,20,Trackchanged)
  cv2.createTrackbar('ErsKerSize', winName, 5, 20, Trackchanged)
  if isinstance(cam, str):
    cv2.createTrackbar('frame', winName, 8, frameNum, Trackchanged)
  while (True):
    THKersize = cv2.getTrackbarPos("THKersize", winName)
    C = cv2.getTrackbarPos("C", winName)
    OpKerSize = cv2.getTrackbarPos("OpKerSize", winName)
    ErsKerSize = cv2.getTrackbarPos("ErsKerSize", winName)
    if THKersize % 2 == 0 or OpKerSize % 2 == 0 or ErsKerSize%2 == 0:
      plt.show
      plt.pause(0.0000001)
      pass
    else:
      if isinstance(cam, str):
        frameIndex = cv2.getTrackbarPos('frame', winName)
        capimg.set(cv2.CAP_PROP_POS_FRAMES, frameIndex-1)
      ret, img = capimg.read()
      if ret == True:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame1 = frame[75:95, 220:420]
        template = cv2.imread('doanday2.jpg', 0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(frame1, template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w - 10, top_left[1] + h)

        frame2 = frame1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()
        #cv2.rectangle(frame1, top_left, bottom_right, (0.0, 255), 2)

        opening_kernel = np.ones((OpKerSize, OpKerSize), np.uint8)

        #if blsize % 2 == 0:
          #pass
        #else:
        th = cv2.adaptiveThreshold(frame2,255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, THKersize, C)
        #ret1, th = cv2.threshold(frame1, C, 255, cv2.THRESH_BINARY_INV)
        #ret2, th = cv2.threshold(frame1, C, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        #th1 = cv2.adaptiveThreshold(frame1,255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, THKersize, C)
        #opening = f.main('dilation',opening_kernel,f.main('erosion',opening_kernel,th))
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, opening_kernel)
        opening1= cv2.morphologyEx(opening,cv2.MORPH_CLOSE,opening_kernel )

        #erosion_kernel = np.zeros((ErsKerSize, ErsKerSize), np.uint8)
        #erosion_kernel[int((ErsKerSize-1)/2),:]=1
        #erosion = f.main('erosion',erosion_kernel,opening)
        #erosion = cv2.erode(opening, erosion_kernel, iterations=1)

        #edges = cv2.Canny(opening, 50, 200, apertureSize=3)
        #edges = opening - erosion;
        edges = cv2.Canny(opening1, 50, 200, apertureSize=3)
        num1, count1 = np.unique(edges, return_counts=True)
        if num1[-1]==255:
          edge_num=count1[-1]
        else:
          edge_num=0
        print(edge_num)
        y[0,frameIndex]=edge_num
        plt.clf()
        plt.plot(x,y,"go")
        plt.grid()

        plt.show
        plt.pause(0.0000001)
        all = cv2.vconcat([th, opening, edges])

        #creat title for each image
        vtitle_size = int((480-all.shape[0])/3)
        htitle_size = all.shape[1]

        th_title = np.zeros((vtitle_size,htitle_size,1),np.uint8)
        th_title[:] = [220]
        cv2.putText(img=th_title,text='Binary image',fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=0,thickness=1,lineType=cv2.LINE_AA,org=(55,75))

        op_title = np.zeros((vtitle_size, htitle_size, 1), np.uint8)
        op_title[:] = [220]
        cv2.putText(img=op_title, text='After opening', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=0, thickness=1,
                    lineType=cv2.LINE_AA, org=(55,75))

        edge_title = np.zeros((vtitle_size, htitle_size, 1), np.uint8)
        edge_title[:] = [220]
        cv2.putText(img=edge_title, text='Edge image', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=0,thickness=1,
                    lineType=cv2.LINE_AA, org=(55,75))
        if edge_num > 45:
          cv2.rectangle(frame, (220, 75), (420, 95), (255, 255, 255), 2)
          cv2.putText(img=frame, text='ERROR', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=255,thickness=2, org=(150, 90))
        else:
          cv2.rectangle(frame, (220, 75), (420, 95), (0, 0, 0), 2)
          cv2.putText(img=frame, text='NORMAL', fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=0, thickness=2, org=(150, 90))
        #all = cv2.vconcat([th_title, th, op_title, opening, edge_title, edges])
        all = cv2.hconcat([frame, cv2.vconcat([th_title, th, op_title, opening, edge_title, edges])])

        cv2.imshow('calib',all)
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
testthreshold(winName='Test trackbar',cam='video_test_day.avi')



