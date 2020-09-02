#Thu vien nay bao gom nhung ham con co lien quan den thuat toan nhan dang
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from scipy.signal import butter,filtfilt

def yolov3_detect(frame):
    #Load yolo
    net = cv2.dnn.readNetFromDarknet("D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3.cfg", "D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3_best.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    classes = []
    with open("D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    #Input to yolo
    height, width, channels = frame.shape
    # Input blob thay vi frame de giam anh huong cua anh sang
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # hien_man_hinh
    detector_idxs = []
    confidences = []
    boxes = []
    for o in outs:
        for detection in o:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0:
                top_left_x = round((detection[0] - detection[2] /2) * width)  # top left x = center_x - width
                top_left_y = round((detection[1] - detection[3] /2) * height)  # top left y = center_y - height
                bottom_right_x = round((detection[0] + detection[2] /2) * width)  # bottom right x = center_x + width
                bottom_right_y = round((detection[1] + detection[3] /2) * height)  # bottom right y = center_y + height
                boxes.append([top_left_x,top_left_y,bottom_right_x,bottom_right_y])
                confidences.append(float(confidence))
                detector_idxs.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0, 0.4)

    for i in indexes:
        i = i[0]
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = boxes[i]
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 250, 0), 2)
        cv2.putText(frame, str(detector_idxs[i]) + '  ' + str(np.round(confidences[i] * 100, 2)) + '%', (top_left_x, top_left_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 1, lineType=cv2.LINE_AA)
        #cv2.imshow('frame',frame)
        #cv2.waitKey(0)
    return frame



def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data,padlen=0)
    return y


#hieu chinh gamma cho mot anh hoac cho folder anh
def gamma_correct(img_dir='',data_dir='',gamma=1,save_dir=''):
    #kiem tra xem img_dir va data_dir co dong thoi ton tai hay khong
    if img_dir and data_dir:
        raise Exception('just declare one directory')

    invGamma = 1.0 / gamma
    #tao lookup table
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    if img_dir:
        origin_img = cv2.imread(img_dir)
        result_img = cv2.LUT(origin_img, table)
        img_diff =  np.hstack([origin_img,result_img])
        cv2.imshow("image difference",img_diff)
        cv2.waitKey(0)
        return result_img
    else:
        img_arr = os.listdir(data_dir)
        for i in img_arr:
            origin_img = cv2.imread(data_dir + '\\' + i)
            result_img = cv2.LUT(origin_img, table)
            img_diff = np.hstack([origin_img, result_img])
            cv2.imshow("image " + i+ " difference", img_diff)
            #cv2.waitKey(0)
            cv2.imwrite(save_dir + '\\' + i, result_img)
    return


def CaptureSample(save_dir='',extention='.jpg',cam=0,heigth=480,width=640,color=1,dmin=20,dmax=60):
    cap=cv2.VideoCapture(cam)
    #cap.set(4,int(heigth) )
    #cap.set(3,int(width))
    if not cap.isOpened():
        print('Could not open camera')
        return
    i = dmin
    while i <= dmax:
        ok,img=cap.read()
        if not ok:
            break
        if color==0:
          img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('captured image',img)
        k = cv2.waitKey(300) & 0xff
        if (k == 27):
            break
        elif (k == ord('s')):
            cv2.destroyAllWindows()
            cv2.imwrite(save_dir + '\\' + str(i) + extention, img)
            '''for j in range(1,11):
                cv2.destroyAllWindows()
                cv2.imwrite(save_dir+'\\'+str(i)+'.'+str(j)+extention,img)'''
            cv2.imshow(str(i)+extention+'  saved',img)
            #f = open(dlink+name+str(i)+'.txt','x+')
            #f.write(str(dmax))
            #f.close()
            k = cv2.waitKey() & 0xff
            cv2.destroyAllWindows()
            if (k == ord('d')):
                os.remove(save_dir + '\\' + str(i) + extention)
                '''for j in range(1,11):
                    os.remove(save_dir+'\\'+str(i)+'.'+str(j)+extention)'''
                cv2.imshow(str(i)+extention+'  deleted',img)
                cv2.waitKey()
                cv2.destroyAllWindows()
            else:
                i += 1
    cap.release()
    cv2.destroyAllWindows()


#Lay du lieu khoang cach va nghich dao do dai box tu data_dir
def getdata4linReg(data_dir='',
                   weights_dir='yolov3_best.weights',
                   cfg_dir='yolov3.cfg',
                   names_dir='obj.names',
                   save_dir=''):
    # array chua cac ten anh trong thu muc co duong dan data_dir
    img_array = os.listdir(data_dir)
    # array chua khoang cach thuc cua moi anh
    dis_array = np.array([])
    # array chua nghich dao do dai bounding box duoc tao tu yolov3
    inv_w_array = np.array([])

    # khoi tao mang yolov3
    net = cv2.dnn.readNetFromDarknet(cfg_dir, weights_dir)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    classes = []
    with open(names_dir, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    for i in img_array:
        frame = cv2.imread(data_dir + '\\' + i)
        img = frame.copy()
        # image = pyramid(img, scaleim)

        height, width, channels = img.shape
        # nhandien
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # hien_man_hinhh
        detector_idxs = []
        confidences = []
        boxes = []
        for o in outs:
            for detection in o:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rec coord
                    x = int(center_x - w / 2)  # top left x
                    y = int(center_y - h / 2)  # top left y

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    detector_idxs.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0, 0.4)
        x = 0
        y = 0
        w = 0
        h = 0
        for j in range(len(boxes)):
            if j in indexes:
                x, y, w, h = boxes[j]
                # Lay data cho mang inv_w_array
                inv_w_array = np.append(inv_w_array, 1/w)
                # Lay data cho mang dis_array
                dis_array = np.append(dis_array, [float(i.split('.')[0])])
                # Luu cac mang duoi dang file .npy
                # print(w)
                #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 250, 0), 2)
                '''cv2.putText(img, str(detector_idxs[j]) + '  ' + str(np.round(confidences[j] * 100, 2)) + '%',
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 1, lineType=cv2.LINE_AA)'''
                # cv2.imshow(i, img)
                # cv2.waitKey(0)
    np.save(save_dir + '\\inv_w_array.npy', inv_w_array)
    np.save(save_dir + '\\dis_array.npy', dis_array)
    print(inv_w_array)
    print(dis_array)
    return


def linear_regression(inv_w_array_dir = '',dis_array_dir = ''):
    # Lay du lieu nghich dao do dai box va khoang cach tu file .npy
    inv_w_array = np.load(inv_w_array_dir)
    dis_array = np.load(dis_array_dir)
    # Ve du lieu truoc khi qua bo loc
    plt.plot(inv_w_array,dis_array, 'o')
    # Ve du lieu sau khi qua bo loc
    filtered_inv_w_array = butter_lowpass_filter(data=inv_w_array,cutoff=0.999,fs=10,order=2)
    #print(filtered_inv_w_array)
    plt.plot(filtered_inv_w_array,dis_array,'o')
    # Tao mo hinh hoi quy
    #a, b = np.polyfit(inv_w_array,dis_array,1)
    inv_w_array = np.reshape(inv_w_array,(-1,1))
    model = LinearRegression().fit(inv_w_array,dis_array)
    a = model.coef_
    b = model.intercept_
    # In ket qua ra man hinh
    print('a = '+str(a)+' b= '+str(b))
    dis_pred_array = a * inv_w_array + b
    filtered_dis_pred_array = a * filtered_inv_w_array + b
    print('mean absolute error = '+str(mean_absolute_error(dis_array,dis_pred_array)))
    print('root mean squared error = ' + str(np.sqrt(mean_squared_error(dis_array, dis_pred_array))))
    print('max error = '+str(max_error(dis_array,dis_pred_array)))
    print('mean absolute error after filtered = '+str(mean_absolute_error(dis_array,filtered_dis_pred_array)))
    print('root mean squared error after filtered = ' + str(np.sqrt(mean_squared_error(dis_array, filtered_dis_pred_array))))
    print('max error after filtered = '+str(max_error(dis_array,filtered_dis_pred_array)))
    # Ve duong hoi quy
    plt.plot(inv_w_array, dis_pred_array)
    plt.xlabel('1/w')
    plt.ylabel('Distance (cm)')
    plt.title("Linear regression")
    plt.show()
    return



# Lay template cho thuat toan matching template
# Thuat toan cho phep xem tung frame va chon ra frame lay template
# Ung voi frame lay template dung chuot de ve
# Rất mất công
# Input: duong dan toi file video
# Output:


'''cv2.imshow("result",gamma_correct(img_dir='D:\\WON\\DO_AN\\Changed_data\\49.jpg',gamma=2))
cv2.waitKey(0)'''
'''getdata4linReg(data_dir='D:\\WON\\DO_AN\\Data\\Distance_estimate',
               weights_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3_best.weights',
               cfg_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3.cfg',
               names_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\obj.names',
               save_dir='D:\\WON\\DO_AN\\Code\\Quan')
time.sleep(1)'''
'''linear_regression(inv_w_array_dir='D:\\WON\\DO_AN\\Code\\Quan\\inv_w_array.npy',
                  dis_array_dir='D:\\WON\\DO_AN\\Code\\Quan\\dis_array.npy')
print(butter_lowpass_filter(data=[0,10],cutoff=1,fs=10,order=5))'''
'''for image in os.listdir('D:\\WON\\DO_AN\\Cho_thay_xem'):
    frame = cv2.imread('D:\\WON\\DO_AN\\Cho_thay_xem\\'+image)
    detected_frame = yolov3_detect(frame)
    cv2.imshow(image,detected_frame)
    cv2.waitKey(0)
    cv2.imwrite('D:\\WON\\DO_AN\\Cho_thay_xem\\'+image,detected_frame)'''


