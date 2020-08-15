import imutils
import cv2
import time
import dlib
import os
import glob
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def Kalman_Filter(last_estimate,mea,err_estimate):
  err_measure=1
  q=0.2
  kalman_gain = err_estimate/(err_estimate + err_measure)
  current_estimate = last_estimate + kalman_gain * (mea - last_estimate)
  err_estimate =  (1.0 - kalman_gain)*err_estimate + (last_estimate-current_estimate)*q
  last_estimate=current_estimate
  return current_estimate,err_estimate
def pyramid(image, scale=1, minSize=(30, 30)):
  while True:
    w = int(image.shape[1] / scale)
    image = imutils.resize(image, width=w)

    if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
      break
    return image
def CaptureSample(link='',name='image',imtype='.png',cam=0,heigth=480,width=640,color=1,timesleep=0.05,i=1):
    cap=cv2.VideoCapture(cam)
    cap.set(4,int(heigth) )
    cap.set(3,int(width))
    if not cap.isOpened():
        print('Could not open camera')
        return
    while(True):
        ok,frame=cap.read()
        if color==0:
          frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if not ok:
            break
        time.sleep(timesleep)
        cv2.imshow('Sample',frame)
        k = cv2.waitKey(10) & 0xff
        if (k == 27):
            break
        elif (k==115):
            cv2.imwrite(link+name+str(i)+imtype,frame)
            i+=1
    cap.release()
    cv2.destroyAllWindows()
def pyramidFolder(nameLinkIn='',nameLinkOut='',nameIm='image',imtype='.png',scale=1,minSize=(30, 30),index=1):
  i=index
  while(True):
    img=cv2.imread(nameLinkIn+nameIm +str(i)+imtype)
    if(img is None):
      return
    else:
      cv2.imwrite(nameLinkOut+nameIm+str(i)+imtype,pyramid(img,scale,minSize))
      i+=1
def training_data(train_xml,test_xml,detectorName,Csvm=5):
  options = dlib.simple_object_detector_training_options()
  options.add_left_right_image_flips = True
  options.C = Csvm
  options.num_threads = 4
  options.be_verbose = True
  dlib.train_simple_object_detector(train_xml, detectorName+".svm", options)
  print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(test_xml, detectorName+".svm")))
def xml_to_csv(path):
  xml_list = []
  for xml_file in glob.glob(path + '/*.xml'):
      tree = ET.parse(xml_file)
      root = tree.getroot()
      for member in root.findall('object'):
          value = (root.find('filename').text,
                   int(root.find('size')[0].text),
                   int(root.find('size')[1].text),
                   member[0].text,
                   int(member[4][0].text),
                   int(member[4][1].text),
                   int(member[4][2].text),
                   int(member[4][3].text)
                   )
          xml_list.append(value)
  column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
  xml_df = pd.DataFrame(xml_list, columns=column_name)
  return xml_df


def  convert_xml_to_csv(inpath,outpath):
    xml_df = xml_to_csv(inpath)
    xml_df.to_csv(outpath, index=None)
    print('Successfully converted xml to csv.')
def Test_model(listNameModel,scaleim,cam):
  detectors=[]
  cap=cv2.VideoCapture(cam)
  for model in listNameModel:
    detectors.append(dlib.fhog_object_detector(model))
  while(1):
    start=time.time()
    ok,img=cap.read()
    image =pyramid(img,scaleim)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors, image, upsample_num_times=0, adjust_threshold=0.0)
    for i in range(len(boxes)):
      if (confidences[i] > 1):
        confidences[i]=1
      elif (confidences[i]< 0.1):
        continue
      face= boxes[i]
      x = int(face.left()*scaleim)
      y =int(face.top()*scaleim)
      w = int((face.right()-face.left())*scaleim)
      h = int((face.bottom()- face.top()) *scaleim)
      cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
      cv2.putText(img, str(detector_idxs[i]) + '  ' + str(np.round(confidences[i]*100,2))+'%', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0),1, lineType=cv2.LINE_AA)
    k = cv2.waitKey(1) & 0xff
    if k == 27 :
        break
    end=time.time()
    cv2.putText(img, 'FPS:' + str(np.round(1/(end-start))), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 50, 150),1, lineType=cv2.LINE_AA)
    cv2.imshow("image", img)
  cap.release()
  cv2.destroyAllWindows()



def SaveVideo(linkName='output',cam=0,fps=10):
  cap = cv2.VideoCapture(cam)
   
  # Check if camera opened successfully
  if (cap.isOpened() == False): 
    print("Unable to read camera feed")
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
   
  # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
  out = cv2.VideoWriter(linkName+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
   
  while(True):
    ret, frame = cap.read()
   
    if ret == True: 
       
      # Write the frame into the file 'output.avi'
      out.write(frame)
   
      # Display the resulting frame    
      cv2.imshow('frame',frame)
   
      # Press Q on keyboard to stop recording
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
    # Break the loop
    else:
      break 
   
  # When everything done, release the video capture and video write objects
  cap.release()
  out.release()
   
  # Closes all the frames
  cv2.destroyAllWindows() 
def Estimate_Distance(listNameModel,scaleim,cam,vector_w):
  detectors=[]
  nearOb=0
  NameOb=''
  Distance=1000
  cap=cv2.VideoCapture(cam)
  for model in listNameModel:
    detectors.append(dlib.fhog_object_detector(model))
  while(1):
    start=time.time()
    ok,img=cap.read()
    if not ok:
      break
    image =pyramid(img,scaleim)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors, image, upsample_num_times=0, adjust_threshold=0.0)
    for i in range(len(boxes)):
      if (confidences[i] > 1):
        confidences[i]=1
      elif (confidences[i]< 0.35):
        continue
      face= boxes[i]
      x = int(face.left()*scaleim)
      y =int(face.top()*scaleim)
      w = int((face.right()-face.left())*scaleim)
      h = int((face.bottom()- face.top()) *scaleim)
      w_0=vector_w[0][detector_idxs[i]]
      w_1=vector_w[1][detector_idxs[i]] 
      w_2=vector_w[2][detector_idxs[i]]
      w_3=vector_w[3][detector_idxs[i]]
      w_4=vector_w[4][detector_idxs[i]]
      if(detector_idxs[i]==0):
        dis=w_0+w_1*x+w_2*y+w_3*w+w_4*h
      elif(detector_idxs[i]==1):
        dis=1/(w_0+w_1*y)
      else:
        dis=w_0+w_1*y 
      if(dis<=Distance):
        nearOb=detector_idxs[i]
        Distance=dis
      cv2.rectangle(img, (x,y), (x+w,y+h), (0, 250, 0), 2)
      cv2.putText(img, str(detector_idxs[i]) + '  ' + str(np.round(confidences[i]*100,2))+'%', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250,0, 0),1, lineType=cv2.LINE_AA)
    k = cv2.waitKey(5) & 0xff
    if k == 27 :
        break
    end=time.time()
    if nearOb==0:
      NameOb='Marker Ball'
    elif nearOb==1:
      NameOb='Ta Chong Rung'
    else:
      NameOb='Khoa do day'
    dis_string=''
    if(Distance==1000):
      dis_string='NaN'
    else:
      dis_string=str(np.round(Distance,2))
    cv2.putText(img, 'FPS:' + str(np.round(1/(end-start))) +'   Name:'+ NameOb +'     Distance:' +dis_string+'cm', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),1, lineType=cv2.LINE_AA)
    cv2.namedWindow('result',cv2.WINDOW_NORMAL)
    cv2.imshow('result', img)
    Distance=1000
  cap.release()
  cv2.destroyAllWindows()
def detect_Line(cam=0,normalValue=40,offset=5):
  cap=cv2.VideoCapture(cam)
  start=0
  N_sum=[]
  time_sum=[]
  N_bad=[]
  time_bad=[]
  time_count=0
  while(True):
    ok,frame=cap.read()
    if not ok:
      break
    frame1 = frame[75:95,220:420].copy() 
    #time.sleep(0.05)
    if start==1:
      checkOk,feat=checkLine(image=frame1,normalValue=40,offset=5)
      print(feat)
      time_count=time_count+1
      N_sum.append(feat)
      time_sum.append(time_count)
      if(checkOk):
        cv2.putText(frame, 'Normal', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (220,75), (420,95), (0,255,0), 2)
      else:
        cv2.putText(frame, 'Error', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (220,75), (420,95), (0,0,255), 2)
        N_bad.append(feat)
        time_bad.append(time_count)
    cv2.imshow('result',frame)
    k = cv2.waitKey(5) & 0xff
    if (k == 27):
      break
    elif (k==115):
      start=1
  fig, axe1 = plt.subplots(1, 1)
  fig.suptitle('Giam sat day', fontsize=15)
  axe1.plot(time_sum,N_sum,'g.',markeredgewidth=3,markersize=3,label='Normal')
  axe1.plot(time_bad,N_bad,'r.',markeredgewidth=3,markersize=3,label='Error')
  plt.xlabel('Thoi gian (ms)')
  plt.ylabel('Tong Pixel')
  axe1.grid(True)
  plt.legend()
  plt.show()
  cap.release()
  cv2.destroyAllWindows()
def checkLine(image,normalValue=40,offset=3):
  term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
  track_window=(0,0,image.shape[1],image.shape[0])
  frame2=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  kernel = np.ones((3,3),np.uint8)
  th = cv2.adaptiveThreshold(frame2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
      cv2.THRESH_BINARY_INV,51,4)
  opening1 = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
  opening = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)
  ret, track_window = cv2.CamShift(opening , track_window, term_crit)
  pts=np.int0(cv2.boxPoints(ret))
  y1=pts[0][0]
  y2=pts[1][0]
  y3=pts[2][0]
  y4=pts[3][0]
  if (not (y1==0 and y2==0 and y3==0 and y4==0)) or (max(y1,y2,y3,y4)>=image.shape[1]-30) or (min(y1,y2,y3,y4)<=30):
    opening[:,0:min(y1,y2,y3,y4)-10]=0
    opening[:,max(y1,y2,y3,y4)+10:image.shape[1]]=0
    resu=opening[:,max(min(y1,y2,y3,y4)-10,0):min(max(y1,y2,y3,y4)+10,image.shape[1])].copy()
    edges = cv2.Canny(opening,50,200,apertureSize=3)
    feat=np.sum(edges/255)
    if((feat<=(normalValue-offset) ) or (feat>=(normalValue+offset))):
      return 0,feat
    else:
      return 1,feat

def combineImage(image1,image2,image,scaleim=1):
  image1=pyramid(image=image1,scale=scaleim)
  image2=pyramid(image=image2,scale=scaleim)
  image=pyramid(image=image,scale=scaleim)
  width=int(image.shape[1])
  half_width=int(image.shape[1]/2)
  width1=int(image1.shape[1])
  width2=int(image2.shape[1])
  image[:,0:half_width]=image1[:,0:width1].copy()
  image[:,half_width:width]=image2[:,0:width2].copy()
  return image
def Demo(listNameModel,scaleim,cam,vector_w,normalValue=40,offset=5):
  detectors=[]
  nearOb=0
  NameOb=''
  Distance=1000
  ob_in_area=0
  sdetect=0
  frball=0
  frta=0
  frkhoa=0
  nhan_dang=0
  term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
  track_window=(0,0,95-75,420-220)
  Image_show=cv2.imread('imagezero.jpg')
  out = cv2.VideoWriter('VIDEO1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,480))
  start=time.time()
  cap=cv2.VideoCapture(cam)
  for model in listNameModel:
    detectors.append(dlib.fhog_object_detector(model))
  while(1):
    ok,frame=cap.read()
    if not ok:
      break
    img=frame.copy()
    image =pyramid(img,scaleim)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    nearOb=-1
    [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors, image, upsample_num_times=0, adjust_threshold=0.0)
    for i in range(len(boxes)):
      nhan_dang=1
      if (confidences[i] > 1):
        confidences[i]=1
      elif (confidences[i]< 0.35):
        continue
      face= boxes[i]
      x = int(face.left()*scaleim)
      y =int(face.top()*scaleim)
      w = int((face.right()-face.left())*scaleim)
      h = int((face.bottom()- face.top()) *scaleim)
      if(((detector_idxs[i]==2) and (y<=200)) or (y<=95)):
        ob_in_area=1
      w_0=vector_w[0][detector_idxs[i]]
      w_1=vector_w[1][detector_idxs[i]] 
      w_2=vector_w[2][detector_idxs[i]]
      w_3=vector_w[3][detector_idxs[i]]
      w_4=vector_w[4][detector_idxs[i]]
      dis1=0
      dis2=0
      dis3=0
      if(detector_idxs[i]==0):
        dis1=w_0+w_1*x+w_2*y+w_3*w+w_4*h
        frball+=1
      elif(detector_idxs[i]==1):
        dis2=1/(w_0+w_1*y)
        frta+=1
      else:
        dis3=w_0+w_1*y 
        frkhoa+=1
      cv2.rectangle(img, (x,y), (x+w,y+h), (0, 250, 0), 2)
      cv2.putText(img, str(detector_idxs[i]) + '  ' + str(np.round(confidences[i]*100,2))+'%', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250,0, 0),1, lineType=cv2.LINE_AA)

    frtotal=max(frball,frta,frkhoa)
    if(frtotal==0):
      nearOb=-1
      Distance=1000
    elif(frtotal==frball):
      nearOb=0
      Distance=dis1
    elif(frtotal==frta):
      nearOb=1
      Distance=dis2
    elif(frtotal==frkhoa):
      nearOb=2
      Distance=dis3
      # dis =max(dis1,dis2,dis3)
      # if(dis<=Distance):
      #   nearOb=detector_idxs[i]
      #   Distance=dis
    if nearOb==0:
      NameOb='Marker Ball'
    elif nearOb==1:
      NameOb='Ta Chong Rung'
    elif nearOb==2:
      NameOb='Khoa do day'
    else:
      NameOb='Nothing'
    dis_string=''
    if(Distance==1000 or nhan_dang==0 or np.round(Distance,2)<21):
      dis_string='Unknow'
    else:
      dis_string=str(np.round(Distance,2))
    end=time.time()
    cv2.putText(img, 'FPS:' + str(np.round(1/(end-start))) +'   Name:'+ NameOb +'     Distance:' +dis_string+'  cm', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),1, lineType=cv2.LINE_AA)
    start=time.time()    

    frame1 = frame[75:95,220:420].copy()
    checkOk,feat=checkLine(image=frame1,normalValue=40,offset=5)
    cv2.rectangle(frame, (220,75), (420,95), (0,255,0), 2) 
    frame2=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3),np.uint8)
    th = cv2.adaptiveThreshold(frame2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,51,4)
    opening1 = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)
    ret, track_window = cv2.CamShift(opening , track_window, term_crit)
    pts=np.int0(cv2.boxPoints(ret))
    y1=pts[0][0]
    y2=pts[1][0]
    y3=pts[2][0]
    y4=pts[3][0]
    if (not (y1==0 and y2==0 and y3==0 and y4==0)) or (max(y1,y2,y3,y4)>=150) or (min(y1,y2,y3,y4)<=50):
      opening[:,0:min(y1,y2,y3,y4)-10]=0
      opening[:,max(y1,y2,y3,y4)+10:200]=0
      resu=opening[:,max(min(y1,y2,y3,y4)-10,0):min(max(y1,y2,y3,y4)+10,200)].copy()
      edges = cv2.Canny(opening,50,200,apertureSize=3)
      feat=np.sum(edges/255)
      print(feat)
      if sdetect==1:
        if ob_in_area==1:
          cv2.putText(frame, 'Obstacle', (70,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0),2, lineType=cv2.LINE_AA)
          cv2.putText(img, 'Lightning rods: Unknow' , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),1, lineType=cv2.LINE_AA)
          cv2.rectangle(frame, (220,75), (420,95), (255,0,0), 2) 
        elif((feat<=37 ) or (feat>=43)):
          cv2.putText(frame, 'Error', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2, lineType=cv2.LINE_AA)
          cv2.putText(img, 'Lightning rods: Error' , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),1, lineType=cv2.LINE_AA)
          cv2.rectangle(frame, (220,75), (420,95),  (0,0,255), 2)
        else:
          cv2.putText(frame, 'Normal', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2, lineType=cv2.LINE_AA)
          cv2.putText(img, 'Lightning rods: Normal' , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),1, lineType=cv2.LINE_AA)
          cv2.rectangle(frame, (220,75), (420,95), (0,255,0), 2) 

    #time.sleep(0.05)
    # if sdetect==1:
    #   if ob_in_area==1:
    #     cv2.putText(frame, 'Obstacle', (70,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0),2, lineType=cv2.LINE_AA)
    #     cv2.putText(img, 'Lightning rods: Unknow' , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),1, lineType=cv2.LINE_AA)
    #     cv2.rectangle(frame, (220,75), (420,95), (255,0,0), 2) 
    #   elif(checkOk):
    #     cv2.putText(frame, 'Normal', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2, lineType=cv2.LINE_AA)
    #     cv2.putText(img, 'Lightning rods: Normal' , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),1, lineType=cv2.LINE_AA)
    #     cv2.rectangle(frame, (220,75), (420,95), (0,255,0), 2) 
    #   else:
    #     cv2.putText(frame, 'Error', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2, lineType=cv2.LINE_AA)
    #     cv2.putText(img, 'Lightning rods: Error' , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),1, lineType=cv2.LINE_AA)
    #     cv2.rectangle(frame, (220,75), (420,95),  (0,0,255), 2) 
    Image_show=combineImage(img,frame,Image_show,scaleim=1)
    cv2.namedWindow('result',cv2.WINDOW_NORMAL)
    cv2.imshow('result', Image_show)
    Distance=1000
    nhan_dang=0
    k = cv2.waitKey(5) & 0xff
    if k == 27 :
      break
    elif (k==115):
      sdetect=1
    elif(k==99):
      ob_in_area=0
      frball=0
      frta=0
      frkhoa=0
    out.write(Image_show)
  cap.release()
  cv2.destroyAllWindows()


  
  
        
      
    
  
    
