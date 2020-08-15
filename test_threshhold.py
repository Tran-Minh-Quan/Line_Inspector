import cv2
import numpy as np
def testthreshold(cam):
  capimg = cv2.VideoCapture(cam)
  if (capimg.isOpened() == False):
    print("Unable to read camera feed")
  if isinstance(cam, str):
    frameNum = int(capimg.get(cv2.CAP_PROP_FRAME_COUNT))


  def Trackchanged(x):
    pass
  cv2.namedWindow('Tracking')
  cv2.createTrackbar("blocksize", "Tracking", 897, 900, Trackchanged)
  cv2.createTrackbar("C", "Tracking", 53,100 , Trackchanged)
  if isinstance(cam, str):
    cv2.createTrackbar('frame', "Tracking", 8, frameNum, Trackchanged)
  while (True):
    blsize = cv2.getTrackbarPos("blocksize", "Tracking")
    constC = cv2.getTrackbarPos("C", "Tracking")
    if isinstance(cam, str):
      frameIndex = cv2.getTrackbarPos('frame', "Tracking")
      capimg.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
    ret, img = capimg.read()
    if ret == True:
      frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      frame1 = frame[75:95, 220:420].copy()
      kernel = np.ones((3, 3), np.uint8)
      if blsize % 2 == 0:
        pass
      else:
        th = cv2.adaptiveThreshold(frame1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY_INV, blsize, constC)
        opening1 = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        opening2 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(opening1, 50, 200, apertureSize=3)
        cv2.rectangle(frame, (220, 75), (420, 95), (0, 0, 255), 2)
        cv2.imshow("op1", opening1)
        cv2.imshow('op2',opening2)
        cv2.imshow('edges',edges)
        cv2.imshow('frame', frame)

      # Press Q on keyboard to stop recording
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Break the loop
    else:
      break

      # When everything done, release the video capture and video write objects
  capimg.release()

  # Closes all the frames
  cv2.destroyAllWindows()
#video2.avi: blocksize = 897, c = 27, canny dùng opening 2
#checkline.avi: blocksize = 897, c = 53, canny dùng opening 1
testthreshold('video2.avi')