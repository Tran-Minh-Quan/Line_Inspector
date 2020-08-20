#Thu vien nay bao gom nhung ham con co lien quan den thuat toan nhan dang
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
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
        k = cv2.waitKey(1) & 0xff
        if (k == 27):
            break
        elif (k == ord('s')):
            for j in range(1,11):
                cv2.destroyAllWindows()
                cv2.imwrite(save_dir+'\\'+str(i)+'.'+str(j)+extention,img)
            cv2.imshow(str(i)+extention+'  saved',img)
            #f = open(dlink+name+str(i)+'.txt','x+')
            #f.write(str(dmax))
            #f.close()
            k = cv2.waitKey() & 0xff
            cv2.destroyAllWindows()
            if (k == ord('d')):
                for j in range(1,11):
                    os.remove(save_dir+'\\'+str(i)+'.'+str(j)+extention)
                cv2.imshow(str(i)+extention+'  deleted',img)
                cv2.waitKey()
                cv2.destroyAllWindows()
            else:
                i += 1
    cap.release()
    cv2.destroyAllWindows()


def linear_regression(data_dir='',weights_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3_best.weights',
                      cfg_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3.cfg',
                      names_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\obj.names'):
    #array chua cac ten anh trong thu muc co duong dan data_dir
    img_array = os.listdir(data_dir)
    #array chua khoang cach thuc cua moi anh
    dis_array = np.array([])
    #array chua do dai bounding box duoc tao tu yolov3
    inv_w_array = np.array([])

    #khoi tao mang yolov3
    net = cv2.dnn.readNetFromDarknet(cfg_dir, weights_dir)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    classes = []
    with open(names_dir, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    for i in img_array:
        frame = cv2.imread(data_dir+'\\'+i)
        img = frame.copy()
        #image = pyramid(img, scaleim)

        height, width, channels = img.shape
        # nhandien
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), True, crop=False)

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
                # tao mang chua cac gia tri do dai bounding box
                x, y, w, h = boxes[j]
                inv_w_array = np.append(inv_w_array,1/w)
                # tao mang chua cac gia tri khoang cach
                dis_array = np.append(dis_array, [float(i.split('.')[0])])
                #print(w)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 250, 0), 2)
                cv2.putText(img, str(detector_idxs[j]) + '  ' + str(np.round(confidences[j] * 100, 2)) + '%',
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 1, lineType=cv2.LINE_AA)
                #cv2.imshow(i, img)
                #cv2.waitKey(0)
    inv_w_array = np.reshape(inv_w_array,(-1,1))
    print(dis_array)
    #print(len(dis_arr))
    print(inv_w_array)
    #print(len(bbwid_arr))
    plt.plot(inv_w_array,dis_array, 'o')
    #a, b = np.polyfit(inv_w_array,dis_array,1)
    model = LinearRegression().fit(inv_w_array,dis_array)
    a = model.coef_
    b = model.intercept_
    print('a = '+str(a)+' b= '+str(b))
    dis_pred_array = a * inv_w_array + b
    print('mean absolute error = '+str(mean_absolute_error(dis_array,dis_pred_array)))
    print('max error = '+str(max_error(dis_array,dis_pred_array)))
    plt.plot(inv_w_array, dis_pred_array)
    plt.xlabel('1/w')
    plt.ylabel('Distance (cm)')
    plt.title("Linear regression")

    plt.show()
    return

linear_regression(data_dir='D:\\WON\\DO_AN\\Changed_data',
                  weights_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3_best.weights',
                      cfg_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3.cfg',
                      names_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\obj.names')

'''cv2.imshow("result",gamma_correct(img_dir='D:\\WON\\DO_AN\\Changed_data\\49.jpg',gamma=2))
cv2.waitKey(0)'''
