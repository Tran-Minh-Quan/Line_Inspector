import cv2
import Sahelpers as Sa
import numpy as np

#Sa.SaveVideo(cam=1,linkName='bonus//bonus2110//bonus3',fps=10)
#Sa.CaptureSample(link='distance_treo2/',name='image',cam=1,color=0,i=0)
#Sa.pyramidFolder(nameLinkIn='newsample/newcam/',nameLinkOut='newsample/newcam/',nameIm='image',scale=2,index=136)
#Sa.training_data('traintacam/traintacam.xml','model/testtacam/testtacam.xml','tacam',Csvm=5)
#Sa.Estimate_Distance(listNameModel=['model/ta.svm'],scaleim=1,cam=1,vector_w= [[ 0.06776287],[-0.00016877]])
#Sa.Estimate_Distance(listNameModel=['model/ta.svm'],scaleim=1,cam=1,vector_w= [[-7.46574639e+01],[ 6.74165985e-02],[ 3.22944320e-01],[ 1.54738448e-01],[ 6.38915972e-02]])
#Sa.Estimate_Distance(listNameModel=['model/ball.svm','model/ta.svm','model/tacam.svm'],scaleim=1,cam=1,vector_w= [[ 7.28016363e+00,0.06776287,9.12270363],[-1.05993695e-04,-0.00016877,0.15939607],[ 1.66308603e-01,0,0],[ 1.26030194e-01,0,0],[-1.44926663e-01,0,0]])
#Sa.Estimate_Distance(listNameModel=['model/tacam.svm'],scaleim=1,cam=1,vector_w= [[41.31076165],[-0.08435581],[ 0.17110782],[-2.52020698],[ 1.94954273]])
Sa.Test_model(listNameModel=['ball.svm'],scaleim=1,cam='video6.avi')
#Sa.detect_Line(cam='bonus/bonus2010/line1.avi')


