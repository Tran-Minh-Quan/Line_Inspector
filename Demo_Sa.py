import cv2
import Sahelpers as Sa
import numpy as np

#Sa.testthreshold(cam=0)

#Sa.Estimate_Distance(listNameModel=['model/ball.svm','model/ta.svm','model/tacam.svm'],scaleim=1,cam=1,vector_w= [[ 7.28016363e+00,0.06776287,9.12270363],[-1.05993695e-04,-0.00016877,0.15939607],[ 1.66308603e-01,0,0],[ 1.26030194e-01,0,0],[-1.44926663e-01,0,0]])
#Sa.detect_Line(cam='bonus/bonus2010/line1.avi')
'''
Sa.Demo(listNameModel=['modelball.svm','modelta.svm','modeltreo.svm'],scaleim=1,cam='video5.avi',\
		vector_w= [[ 7.28016363e+00,0.06776287,9.12270363],\
					[-1.05993695e-04,-0.00016877,0.15939607],\
					[ 1.66308603e-01,0,0],\
					[ 1.26030194e-01,0,0],\
					[-1.44926663e-01,0,0]])

'''
#Sa.CaptureSample( imglink='C:/Users/PC/Desktop/test/img',dlink='C:/Users/PC/Desktop/test/dis/',name='image',cam=0,color=1,i=1,dmax=10)
#Sa.AutoCaptureSample(link='D:\WON\DO_AN\Code\\train\\', name='image',imNum=50,captureTime=1/30,imtype='.png', cam='http://192.168.1.3:8080/video', heigth=480, width=640, color=1)
import An.apply_yolo as An

An.Demo(scaleim=1,cam='video2.avi',\
		vector_w= [[ 7.28016363e+00,0.06776287,9.12270363],\
					[-1.05993695e-04,-0.00016877,0.15939607],\
					[ 1.66308603e-01,0,0],\
					[ 1.26030194e-01,0,0],\
					[-1.44926663e-01,0,0]])
