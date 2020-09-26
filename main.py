import cv2
from Quan import QuanLib
from Huy.checkingline import Line
import numpy as np
import sys
import time
from Tho.Main_lib import distance_calc as Tho

# Mở camera
cap = cv2.VideoCapture(1)
if cap.isOpened() == False:
    sys.exit("Unable to read camera feed")

# Tạo mô hình nhận dạng
model = QuanLib.Yolov3(weights_dir='D:\\WON\\DO_AN\\Model\\yolov3_best.weights',
                        cfg_dir='D:\\WON\\DO_AN\\Model\\yolov3.cfg',
                          names_dir='D:\\WON\\DO_AN\\Model\\obj.names')

# Tạo bộ lọc
filter = QuanLib.butter_lowpass_filter(order=2, cutoff=0.7, fs=10)

# Khởi tạo bộ ước lượng khoảng cách
Circle_based_estimate_1 = Tho.CircleDistance(50, 5000000, 1, 50, 867.7887424687945, -0.18242145320198588)
# Khai báo các biến
ob_in_area = 0 # Biến trạng thái, cho biết vật cản có nằm trong vùng xét dây hay không
inv_w_array = [0, 0, 0] # Ma trận lưu 3 giá trị nghịch đảo độ dài box gần nhất
pre_time = time.time() # Biến tạm thời gian
a = 4132.45382956        # a,b là hệ số của phương trình hồi quy tuyến tính
b = 2.4475897335913714   # y = ax + b
img_out = np.zeros((480,640,3))
while True:
    # Đọc frame
    ok, frame = cap.read()
    if not ok:
        sys.exit("Cannot grab the frame or reached the end of the video")

    # Nhận dạng vật cản
    model.detect(frame)

    # Phát hiện đoạn dây hỏng
    line = Line(frame=frame, ROI=[220, 75, 420, 95])
    line.detect_error(template_dir='D:\\WON\\DO_AN\\Code\\Huy\\template.jpg',min_thres=36, max_thres=48)

    # Ước lượng khoảng cách
    if model.box_width != None:
        if model.box_width != 0:
            inv_w_array = inv_w_array[1:3] + [1 / model.box_width]
            # print(inv_w_array)
            #Distance = round(a * filter.output(data=inv_w_array)[2] + b,1)
            Distance, img_out, _ = Circle_based_estimate_1.calculate(frame, (model.top_left_y, model.top_left_x), (model.bottom_right_y, model.bottom_right_x),
                                                         0.2, mode=1, object_width=10)
    else:
        Distance = ' ?  '
        pass
    # Hiển thị kết quả nhận dạng vật cản
    if [model.top_left_x,model.top_left_y,model.bottom_right_x,model.bottom_right_y] != [None,None,None,None]:
        if model.name == 'ball':
            cv2.rectangle(frame, (model.top_left_x, model.top_left_y), (model.bottom_right_x, model.bottom_right_y), (128, 0, 128), 2)
        elif model.name == 'damper':
            cv2.rectangle(frame, (model.top_left_x, model.top_left_y), (model.bottom_right_x, model.bottom_right_y), (0, 80, 0), 2)
        elif model.name == 'clamp':
            cv2.rectangle(frame, (model.top_left_x, model.top_left_y), (model.bottom_right_x, model.bottom_right_y), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (model.top_left_x, model.top_left_y), (model.bottom_right_x, model.bottom_right_y), (0, 255, 255), 2)

    # Hiển thị kết quả phát hiện dây lỗi
    if model.top_left_y != None:
        if (model.top_left_y <= line.top_left_y and model.bottom_right_y >= line.bottom_right_y)\
                or (model.top_left_y <= line.bottom_right_y and model.top_left_y > line.top_left_y)\
                or (model.bottom_right_y >= line.top_left_y and model.bottom_right_y < line.bottom_right_y):
            ob_in_area = 1
    if ob_in_area == 0:
        if line.status == 'Normal':
            '''cv2.putText(frame, 'Normal', (line.top_left_x - 130, line.top_left_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2, lineType=cv2.LINE_AA)'''
            cv2.rectangle(frame, (line.top_left_x, line.top_left_y), (line.bottom_right_x, line.bottom_right_y),
                          (0, 255, 0), 2)
        else:
            '''cv2.putText(frame, 'Error', (line.top_left_x - 130, line.top_left_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2, lineType=cv2.LINE_AA)'''
            cv2.rectangle(frame, (line.top_left_x, line.top_left_y), (line.bottom_right_x, line.bottom_right_y),
                          (0, 0, 255), 2)
    # Hiển thị bảng kết quả
    info_panel = np.zeros((98,640,3), np.uint8)
    cycle_time = time.time() - pre_time
    if cycle_time != 0:
        FPS = round(1/cycle_time)
    pre_time = time.time()
    line_1 = 'FPS: ' + str(FPS) + '          ' + 'Cycle time: ' + str(round(cycle_time*1000,3)) + 'ms'
    line_2 = 'Name: ' + model.name + '     ' + 'Confidence: ' + str(round(model.confidence * 100, 1)) + ' %'
    line_3 = 'Distance: ' + str(Distance) + 'cm' + ' '
    if ob_in_area == 1:
        line_3 = line_3 + 'Line status:   ?'
    else:
        line_3 = line_3 + 'Line status: ' + line.status
    cv2.putText(info_panel, line_1, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(info_panel, line_2, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(info_panel, line_3, (5, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    # Hiển thị toàn bộ
    result_image = cv2.vconcat([info_panel, frame])
    cv2.imshow('result',result_image)
    cv2.imshow('circle',img_out)

    # Thao tác bàn phím
    toggle = cv2.waitKey(1)
    if toggle == ord('c'): # Đã vượt qua vật cản, continue
        ob_in_area = 0
    if toggle == 27:
        break
cap.release()
cv2.destroyAllWindows()



'''
Sa.Demo(listNameModel=['modelball.svm','modelta.svm','modeltreo.svm'],scaleim=1,cam='video5.avi',\
		vector_w= [[ 7.28016363e+00,0.06776287,9.12270363],\
					[-1.05993695e-04,-0.00016877,0.15939607],\
					[ 1.66308603e-01,0,0],\
					[ 1.26030194e-01,0,0],\
					[-1.44926663e-01,0,0]])

'''
#QuanLib.CaptureSample(save_dir='D:\\WON\\DO_AN\\Data\\Distance_estimate',cam=1,color=1,dmin=20, dmax=100)
#Sa.AutoCaptureSample(link='D:\WON\DO_AN\Code\\train\\', name='image',imNum=50,captureTime=1/30,imtype='.png', cam='http://192.168.1.3:8080/video', heigth=480, width=640, color=1)
'''import An.apply_yolo as An

An.Demo(scaleim=1,cam='D:\\WON\\DO_AN\\Code\\Video\\video2.avi',\
		vector_w= [[ 7.28016363e+00,0.06776287,9.12270363],\
					[-1.05993695e-04,-0.00016877,0.15939607],\
					[ 1.66308603e-01,0,0],\
					[ 1.26030194e-01,0,0],\
					[-1.44926663e-01,0,0]])'''
