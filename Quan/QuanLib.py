#Thu vien nay bao gom nhung ham con co lien quan den thuat toan nhan dang
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from scipy.signal import butter,filtfilt


# Class bao gồm những lệnh liên quan yolo
class Yolo:
    def __init__(self, cfg_dir, weights_dir, names_dir):
        self.cfg_dir = cfg_dir
        self.weights_dir = weights_dir
        self.names_dir = names_dir
        self.net = cv2.dnn.readNetFromDarknet(self.cfg_dir,self.weights_dir)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.classes = []
        with open(names_dir, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self,frame):
        frame_height, frame_width, frame_channels = frame.shape
        # Input blob thay vi frame de giam anh huong cua anh sang
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
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
                    top_left_x = round((detection[0] - detection[2] / 2) * frame_width)  # top left x = center_x - width
                    top_left_y = round((detection[1] - detection[3] / 2) * frame_height)  # top left y = center_y - height
                    bottom_right_x = round((detection[0] + detection[2] / 2) * frame_width)  # bottom right x = center_x + width
                    bottom_right_y = round((detection[1] + detection[3] / 2) * frame_height)  # bottom right y = center_y + height
                    boxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
                    confidences.append(float(confidence))
                    detector_idxs.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0, 0.4)

        results = []
        for i in indexes:
            i = i[0]
            obstacle = {}
            obstacle['name'] = self.classes[detector_idxs[i]]
            obstacle['confidence'] = round(confidences[i]*100)
            obstacle['top left'] = (boxes[i][0],boxes[i][1])
            obstacle['bottom right'] = (boxes[i][2],boxes[i][3])
            obstacle['width'] = boxes[i][2] - boxes[i][0]
            results.append(obstacle)
        return results


class butter_lowpass_filter:
    def __init__(self, order, cutoff, fs):
        self.order = order
        self.cutoff = cutoff
        self.fs =fs
        self.nyq = 0.5 * self.fs
        self.normal_cutoff = self.cutoff / self.nyq
        # Get the filter coefficients
        self.b, self.a = butter(self.order, self.normal_cutoff, btype='low', analog=False)
    def output(self,data):
        return filtfilt(self.b, self.a, data, padlen=0)


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
        for i, img_name in enumerate(img_arr,1):
            origin_img = cv2.imread(data_dir + '\\' + img_name)
            result_img = cv2.LUT(origin_img, table)
            print('Number of image processed: '+str(i)+'/'+str(len(img_arr)))
            # img_diff = np.hstack([origin_img, result_img])
            # cv2.imshow("image " + img_name+ " difference", img_diff)
            # cv2.waitKey(0)
            cv2.imwrite(save_dir + '\\' + img_name, result_img)
    return


def CaptureSample(save_dir='',extention='.jpg',cam=0,heigth=480,width=640,color=1,dmin=20,dmax=60):
    cap = cv2.VideoCapture(cam)
    cap.set(4,int(heigth) )
    cap.set(3,int(width))
    if not cap.isOpened():
        sys.exit('Could not open camera')
    i = dmin
    while i <= dmax:
        ok,frame = cap.read()
        if not ok:
            break
        if color==0:
          frame = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',frame)
        k = cv2.waitKey(300) & 0xff
        if (k == 27):
            break
        elif (k == ord('s')):
            cv2.destroyAllWindows()
            cv2.imwrite(save_dir + '\\' + str(i) + extention, frame)
            '''for j in range(1,11):
                cv2.destroyAllWindows()
                cv2.imwrite(save_dir+'\\'+str(i)+'.'+str(j)+extention,frame)'''
            cv2.imshow(str(i)+extention+'  saved',frame)
            #f = open(dlink+name+str(i)+'.txt','x+')
            #f.write(str(dmax))
            #f.close()
            k = cv2.waitKey() & 0xff
            cv2.destroyAllWindows()
            if (k == ord('d')):
                os.remove(save_dir + '\\' + str(i) + extention)
                '''for j in range(1,11):
                    os.remove(save_dir+'\\'+str(i)+'.'+str(j)+extention)'''
                cv2.imshow(str(i)+extention+'  deleted',frame)
                cv2.waitKey()
                cv2.destroyAllWindows()
            else:
                i += 1
    cap.release()
    cv2.destroyAllWindows()


# Lấy mẫu ảnh
# Nhấp trái chuột vào hình để xác nhận lấy frame hiện tại
# Nhấp trái chuột vào hình lần nữa để quay lại chế độ livestream
# Nhấp phải chuột vào hình  để xóa hình mới lưu
# Nhấp phải chuột vào hình lần nữa để xóa hình đã lưu trước đó
# Input:
# save_dir: địa chỉ thư mục lưu ảnh, để dưới dạng '\' cũng được
# name: tên ảnh
# beginIndex = số thứ tự ảnh bắt đầu
# imgPerIdxNum: số ảnh mỗi index, số thứ tự các ảnh này được biểu diễn bởi sub_index
# extension: đuôi ảnh, vd '.jpg', '.png'
# cấu trúc tên file: name + '_' + index + '_' + sub_index + extension
# cam: nếu cam là số thì là chọn camera để lấy mẫu, nếu cam là str thì là đường dẫn tới video dùng để lấy mẫu
# height: chiều cao ảnh
# width: chiều dài ảnh'''
def getSampleImage(save_dir, name, beginIndex=20, imgPerIdxNum = 0, extention='.jpg', cam=1, height=480, width=640):
  # Capture frame
  cap = cv2.VideoCapture(cam)
  # Kiểm tra xem lấy được frame chưa
  if not cap.isOpened():
    sys.exit('Could not open camera')
  # Chỉnh lại kích thước ảnh
  cap.set(4, int(height))
  cap.set(3, int(width))
  # Tạo title lúc mới khởi động
  title = np.zeros((58, 640, 3), dtype=np.uint8)
  cv2.putText(title, 'Right click to save', (105, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2,
              lineType=cv2.LINE_AA)
  # Khởi tạo các biến
  global isCapturing, index, sub_index, frame #Khai báo biến toàn cục để sử dụng chung với hàm con
  index = beginIndex
  sub_index = 1
  isCapturing = False

  # Hàm gọi tới khi nhấn chuột:
  def mouseCallback(event, x, y, flags, param):
    # Khai báo biến toàn cục để sử dụng chung với hàm mẹ
    global isCapturing, index, sub_index, frame
    # Thao tác khi nhấp trái chuột
    if event == cv2.EVENT_LBUTTONDOWN:
      # Đảo trạng thái của cờ cho biết đang lấy hình hay không
      isCapturing = not isCapturing
      # Thao tác khi đang lấy hình
      if isCapturing:
        # Tăng chỉ số khoảng cách thêm 1
        index += 1
        # Nếu mỗi khoảng cách 1 ảnh
        if imgPerIdxNum == 1:
          ok, frame = cap.read()
          title = np.zeros((58, 640, 3), dtype=np.uint8)
          imageName = name + '_' + str(index-1) + extention
          cv2.putText(title, imageName + ' saved', (105, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2,
                      lineType=cv2.LINE_AA)
          cv2.imwrite(save_dir + '/' + imageName, frame)
          cv2.imshow('frame', cv2.vconcat([title, frame]))
        # Nếu mỗi khoảng cách lấy nhiều ảnh, tự động lấy imgPerIdxNum liên tiếp, cách nhau 0.5s
        else:
          for sub_index in range(1,imgPerIdxNum+1):
            ok, frame = cap.read()
            title = np.zeros((58, 640, 3), dtype=np.uint8)
            imageName = name + '_' + str(index-1) + '_' + str(sub_index) + extention
            cv2.putText(title, imageName + ' saved', (85, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2,
                        lineType=cv2.LINE_AA)
            cv2.imwrite(save_dir + '/' + imageName, frame)
            cv2.imshow('frame', cv2.vconcat([title, frame]))
            cv2.waitKey(500)
    # Thao tác khi nhấp phải chuột
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Nếu mỗi khoảng cách 1 ảnh
        if imgPerIdxNum == 1:
          title = np.zeros((58, 640, 3), dtype=np.uint8)
          imageName = name + '_' + str(index-1) + extention
          cv2.putText(title, imageName + ' deleted', (105, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2,
                      lineType=cv2.LINE_AA)
          deleted_frame = cv2.imread(save_dir + '/' + imageName)
          cv2.imshow('frame', cv2.vconcat([title, deleted_frame]))
          os.remove(save_dir + '/' + imageName)
        # Nếu mỗi khoảng cách lấy nhiều ảnh, tự động xóa imgPerIdxNum liên tiếp, hiển thị ảnh sẽ xóa cách nhau 0.5s
        else:
          for sub_index in reversed(range(1, imgPerIdxNum + 1)):
            title = np.zeros((58, 640, 3), dtype=np.uint8)
            imageName = name + '_' + str(index-1) + '_' + str(sub_index) + extention
            cv2.putText(title, imageName + ' deleted', (65, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2,
                        lineType=cv2.LINE_AA)
            deleted_frame = cv2.imread(save_dir + '/' + imageName)
            cv2.imshow('frame', cv2.vconcat([title,deleted_frame]))
            cv2.waitKey(500)
            os.remove(save_dir + '/' + imageName)
        index -= 1
    return
  # Tạo cửa sổ hiển thị và cài đặt ngắt lên cửa sổ đó
  cv2.namedWindow('frame')
  cv2.setMouseCallback('frame', mouseCallback)

  # Vòng lặp chính
  while True:
    # Nếu không đang lấy frame thì lấy và hiển thị frame mới
    if not isCapturing:
      ok, frame = cap.read()
      if not ok:
        break
      cv2.imshow('frame', cv2.vconcat([title,frame]))
    # Nếu đang lấy frame thì giữ cửa sổ hiển thị
    key = cv2.waitKey(1)
    # Ấn esc để thoát
    if key == 27:
      break

  # Tắt cam và đóng hết các cửa sổ khi thoát
  cap.release()
  cv2.destroyAllWindows()
  return


# Dùng để lấy video mẫu
# Nhấp trái chuột vào màn hình để bắt đầu ghi hình
# Nhấp trái chuột vào màn hình lần nữa để tạm dừng, các frame hình từ lúc này sẽ không được ghi lại
# Nhấp trái chuột vào màn hình lần nữa để tiếp tục ghi hình
# Nhấp phải chuột vào màn hình để dừng ghi hình và lưu video hiện tại
# Nhấn esc để thoát
# Input:
# cam: chọn camera
# save_dir: đường dẫn đến thư mục lưu video
# name: tên video
# beginIndex: số thứ tự bắt đầu của video
# extension: đuôi video
# cấu trúc tên = name + '_' + index + extension
# height: chiều dọc video
# width: chiều ngang video
def getSampleVideo(cam, save_dir,name,beginIndex=1,extension='.avi',height=480,width=640):
  # Mở camera
  cap = cv2.VideoCapture(cam)
  if not cap.isOpened():
    print('Unable to open camera')
  # Cài đặt kích thước frame ảnh camera lấy
  cap.set(4, int(height))
  cap.set(3, int(width))
  # Khởi tạo bộ lấy video
  fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
  global isRecording, paused, video_writer, index
  index = beginIndex
  video_dir = save_dir + '/' + name + '_' + str(index) + extension
  video_writer = cv2.VideoWriter(video_dir,fourcc,30,(width,height))
  # Khởi tạo các biến và title dùng để hiển thị
  isRecording, paused = False, False
  default_title = np.zeros((58, 640, 3), dtype=np.uint8)
  cv2.putText(default_title, '|> Right click to record', (35, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2,
              lineType=cv2.LINE_AA)
  recording_title = np.zeros((58, 640, 3), dtype=np.uint8)
  cv2.putText(recording_title, 'Recording...', (180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2,
              lineType=cv2.LINE_AA)
  paused_title = np.zeros((58, 640, 3), dtype=np.uint8)
  cv2.putText(paused_title, '| | Paused', (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2,
              lineType=cv2.LINE_AA)
  # Tạo hàm gọi đến khi click chuột
  def mouseCallback(event, x, y, flags, param):
    global isRecording, paused, video_writer, index
    if event == cv2.EVENT_LBUTTONDOWN:
      if not isRecording:
        isRecording = True
      else:
        paused = not paused
    if event == cv2.EVENT_RBUTTONDOWN:
      index += 1
      isRecording, paused = False, False
      video_writer.release()
      video_dir = save_dir + '/' + name + '_' + str(index) + extension
      fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
      video_writer = cv2.VideoWriter(video_dir, fourcc, 30, (width, height))
    return
  # Tạo cửa sổ hiển thị và cài đặt ngắt lên cửa sổ đó
  winName = 'frame'
  cv2.namedWindow(winName)
  cv2.setMouseCallback(winName,mouseCallback)
  # Main loop
  while True:
    _, frame = cap.read()
    if paused:
      cv2.imshow(winName, cv2.vconcat([paused_title, frame]))
    elif isRecording:
      video_writer.write(frame)
      cv2.imshow(winName, cv2.vconcat([recording_title, frame]))
    else:
      cv2.imshow(winName, cv2.vconcat([default_title, frame]))
    if cv2.waitKey(1) == 27:
      break
  # Tắt cam, xóa bộ tạo video, xóa video thừa
  cap.release()
  video_writer.release()
  cv2.destroyAllWindows()
  os.remove(save_dir + '/' + name + '_' + str(index) + extension)
  return


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
    # array chua nghich dao do dai bounding box duoc tao tu yolo
    inv_w_array = np.array([])

    # khoi tao mang yolo
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
    inv_w_array = np.loadtxt(inv_w_array_dir)
    dis_array = np.loadtxt(dis_array_dir)
    # Ve du lieu truoc khi qua bo loc
    plt.plot(inv_w_array,dis_array, 'o')
    # Ve du lieu sau khi qua bo loc
    #filtered_inv_w_array = butter_lowpass_filter(cutoff=0.999,fs=10,order=2).output(data = inv_w_array)
    #print(filtered_inv_w_array)
    #plt.plot(filtered_inv_w_array,dis_array,'o')
    # Tao mo hinh hoi quy
    #a, b = np.polyfit(inv_w_array,dis_array,1)
    inv_w_array = np.reshape(inv_w_array,(-1,1))
    model = LinearRegression().fit(inv_w_array,dis_array)
    a = model.coef_
    b = model.intercept_
    # In ket qua ra man hinh
    print('a = '+str(a.item())+' b= '+str(b))
    dis_pred_array = a * inv_w_array + b
    #filtered_dis_pred_array = a * filtered_inv_w_array + b
    print('mean absolute error = '+str(mean_absolute_error(dis_array,dis_pred_array)))
    print('root mean squared error = ' + str(np.sqrt(mean_squared_error(dis_array, dis_pred_array))))
    print('max error = '+str(max_error(dis_array,dis_pred_array)))
    # print('mean absolute error after filtered = '+str(mean_absolute_error(dis_array,filtered_dis_pred_array)))
    # print('root mean squared error after filtered = ' + str(np.sqrt(mean_squared_error(dis_array, filtered_dis_pred_array))))
    # print('max error after filtered = '+str(max_error(dis_array,filtered_dis_pred_array)))
    # Ve duong hoi quy
    plt.plot(inv_w_array, dis_pred_array)
    plt.xlabel('1/w')
    plt.ylabel('Distance (cm)')
    plt.title("Linear regression")
    plt.show()
    return


# Chú thích cho get_template_tool(...):
# Chức năng: Lấy template cho thuật toán template matching
# Thuật toán cho phép xem từng frame của video và chọn ra frame lấy template
# Điều chỉnh tọa độ khung hình chữ nhật bao quanh template bằng trackbar
# Ảnh bên trái là frame, ảnh bên phải là template
# Input: đường dẫn tới video và thư mục lưu template,tên template, ROI
# ROI = [top_left_x,top_left_y, bottom_right_x, bottom_right_y]
# Output: template
def getTemplate(video_dir, save_dir,template_name, ROI):
    cap = cv2.VideoCapture(video_dir)
    frameNum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x = np.array([np.arange(frameNum + 1)])
    y = np.zeros((1, frameNum + 1), np.uint8)

    def Trackchanged(x):
        pass
    winName = 'Get template tool'
    cv2.namedWindow(winName)
    cv2.createTrackbar('top_left_x', winName, ROI[0], width, Trackchanged)
    cv2.createTrackbar('bottom_right_x', winName, ROI[2], width, Trackchanged)
    cv2.createTrackbar('frame', winName, 1, frameNum-1, Trackchanged)

    while (True):
        # Cập nhật vị trí trackbar
        top_left_x = cv2.getTrackbarPos('top_left_x', winName)
        bottom_right_x = cv2.getTrackbarPos('bottom_right_x', winName)
        frameIndex = cv2.getTrackbarPos('frame', winName)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        frame_ok, frame = cap.read()
        if frame_ok:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            blank_img_vsize = (height - (ROI[3]-ROI[1])) // 2
            blank_img_hsize = bottom_right_x - top_left_x
            blank_img = np.zeros((blank_img_vsize, blank_img_hsize, 1), np.uint8)
            blank_img[:] = [220] #chọn màu cho blank_img
            template = gray_frame[ROI[1]:ROI[3],top_left_x:bottom_right_x].copy()
            vconcat = cv2.vconcat([blank_img,template,blank_img])
            cv2.rectangle(gray_frame, (top_left_x, ROI[1]), (bottom_right_x, ROI[3]), (0, 0, 0), 1)
            display_img = cv2.hconcat([gray_frame, cv2.vconcat([blank_img,template,blank_img])])

            cv2.imshow(winName, display_img)
            toggle = cv2.waitKey(1)
            if toggle == ord('s'):
                cv2.imwrite(save_dir+'\\'+template_name,template)
                cv2.destroyAllWindows()
                break
            if toggle == 27: #esc
                cv2.destroyAllWindows()
                break
    return


def cvtImages2Video(img_folder_dir,save_dir,video_name):
    imgs = [img for img in os.listdir(img_folder_dir) if img.endswith('.jpg')]
    # Sắp xếp lại các ảnh theo thứ tự tăng dần khoảng cách
    distances = [int(img.split('.')[0].split('_')[1]) for img in imgs]
    zipped_lists = zip(distances, imgs)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    distances, imgs = [list(tuple) for tuple in tuples]
    # Lấy kích thước frame và tạo bộ ghi hình
    frame = cv2.imread(os.path.join(img_folder_dir, imgs[0]))
    height, width, channels = frame.shape
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter(os.path.join(save_dir,video_name), fourcc, 30, (width, height))
    print(imgs)
    for img in imgs:
        video.write(cv2.imread(os.path.join(img_folder_dir, img)))
    video.release()


def modelTest(img_dir,mask_dir=None,background=1,brightness=1,blur=1,shift=1,rotate=1):
    model = Yolo(weights_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3_best.weights',
                           cfg_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3.cfg',
                           names_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\obj.names')
    img = cv2.imread(img_dir)
    height, width, channels = img.shape
    def Trackchanged(x):
        pass

    winName = 'Model test'
    cv2.namedWindow(winName)
    cv2.createTrackbar('BG', winName, 0, 1, Trackchanged)
    cv2.createTrackbar('Brightness', winName, 1, 100, Trackchanged)
    cv2.createTrackbar('Blur', winName, 1,100, Trackchanged)
    cv2.createTrackbar('Shift_x', winName, 1, width, Trackchanged)
    cv2.createTrackbar('Shift_y', winName, 1, height, Trackchanged)
    cv2.createTrackbar('Rotate', winName, 1,360, Trackchanged)

    while True:
        bg_flag = cv2.getTrackbarPos('BG', winName)
        gamma = cv2.getTrackbarPos('Brightness', winName)
        ksize = cv2.getTrackbarPos('Blur', winName)
        shift_x = cv2.getTrackbarPos('Shift_x', winName)
        shift_y = cv2.getTrackbarPos('Shift_y', winName)
        angle = cv2.getTrackbarPos('Rotate', winName)
        try:
            blurred_img = cv2.GaussianBlur(img.copy(), (ksize, ksize), 0)
            results = model.detect(blurred_img)
            for result in results:
                cv2.putText(blurred_img, result['name']+'    '+str(result['confidence'])+'%',(result['top left'][0],result['top left'][1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, lineType=cv2.LINE_AA)
                cv2.rectangle(blurred_img, result['top left'], result['bottom right'], (0, 255, 0), 1)
        except:
            pass
        cv2.imshow('result', blurred_img)
        cv2.waitKey(1)




# modelTest(r'D:\WON\DO_AN\Data\Training\Lan1\Damper\Images\gamma_corrected_no_repeat\damper_20_1.jpg')
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

# getTemplate(video_dir='D:\\WON\\DO_AN\\Code\\Huy\\video_test_day.avi',
#                   save_dir='D:\\WON\\DO_AN\\Code\\Huy',
#                   template_name='quan1_template.jpg',
#                   ROI=[220, 75, 420, 95])

# CaptureSample('D:\\WON\\DO_AN\\Data', '.jpg', 0, 480, 640, 1, 20, 60)
# getSampleImage(save_dir='D:\WON\DO_AN\Data\Distance\Damper_3',name='damper', beginIndex=20,
#         imgPerIdxNum = 5, extention='.jpg', cam=1, height=480, width=640)
# getSampleVideo(cam=1,save_dir='D:\WON\DO_AN\Data\Training',name='damper',extension='.avi',height=480,width=640)
# gamma_correct(img_dir='D:\WON\DO_AN\Data\Distance\Damper_2\damper_20_1.jpg',data_dir='',gamma=4,save_dir='D:\WON\DO_AN\Data\Distance\Damper_2\damper_20_1_corrected.jpg')
# cvtImages2Video(img_folder_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\No_repeat_images',save_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp',video_name='clamp.mp4')


