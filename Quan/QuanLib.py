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
from scipy.signal import butter, filtfilt
import math
import random
import albumentations as a


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
                if confidence > 0.5:
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


def testSegmentColor(img_or_video_dir):
    if not isinstance(img_or_video_dir,str):
        sys.exit('Ngo vao la duong dan den hinh anh hoac video .avi')
    winName = 'testSegmentColor'
    cv2.namedWindow(winName)
    def callback(x):
        pass
    cv2.createTrackbar('lowHue', winName, 103, 179, callback)
    cv2.createTrackbar('lowSat', winName, 220, 255, callback)
    cv2.createTrackbar('lowVal', winName, 243, 255, callback)
    cv2.createTrackbar('highHue', winName, 103, 179, callback)
    cv2.createTrackbar('highSat', winName, 220, 255, callback)
    cv2.createTrackbar('highVal', winName, 243, 255, callback)
    if img_or_video_dir.endswith('.avi'):
        cap = cv2.VideoCapture(img_or_video_dir)
        frameNum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.createTrackbar('frame', winName, 1, frameNum - 1, callback)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        if img_or_video_dir.endswith('.avi'):
            frameIndex = cv2.getTrackbarPos('frame', winName)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
            frame_ok, frame = cap.read()
        else:
            frame = cv2.imread(img_or_video_dir)
        lowHue = cv2.getTrackbarPos('lowHue', winName)
        lowSat = cv2.getTrackbarPos('lowSat', winName)
        lowVal = cv2.getTrackbarPos('lowVal', winName)
        highHue = cv2.getTrackbarPos('highHue', winName)
        highSat = cv2.getTrackbarPos('highSat', winName)
        highVal = cv2.getTrackbarPos('highVal', winName)

        frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame_hsv,(lowHue,lowSat,lowVal),(highHue,highSat,highVal))

        mask_resized = cv2.resize(mask,(640,480),interpolation=cv2.INTER_NEAREST)
        mask_bgr = cv2.cvtColor(mask_resized,cv2.COLOR_GRAY2BGR)
        org = cv2.imread(r'D:\WON\DO_AN\Data\Training\Lan1\Damper\Images\gamma_corrected_no_repeat\damper_20_1.jpg')
        fg = cv2.bitwise_and(org, 255 - mask_bgr)
        # cv2.imwrite(r'D:\WON\DO_AN\Data\Training\Lan1\Damper\Images\Mask\damper_20_1_mask.png',255 - mask_bgr)
        # print(frame_hsv)
        # print(np.where((mask_bgr > 0) & (mask_bgr < 255)))

        # cv2.imshow('mask',mask)
        # cv2.imshow('frame',frame)
        cv2.imshow('fg',fg)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


def modelTest(img_dir,bbox_dir,mask_dir):
    model = Yolo(weights_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3_best.weights',
                           cfg_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3.cfg',
                           names_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\obj.names')
    img = cv2.imread(img_dir)
    height, width, channels = img.shape
    def Trackchanged(x):
        pass

    winName = 'Model test'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winName, 500, 500)
    cv2.createTrackbar('randBG', winName, 0, 1, Trackchanged)
    cv2.createTrackbar('Gamma', winName, 30, 100, Trackchanged)
    cv2.createTrackbar('Blur', winName, 1,100, Trackchanged)
    cv2.createTrackbar('Shift_x', winName, width, width*2, Trackchanged)
    cv2.createTrackbar('Shift_y', winName, height, height*2, Trackchanged)
    cv2.createTrackbar('Scale',winName,10,100,Trackchanged)
    cv2.createTrackbar('Rotate', winName, 180,360, Trackchanged)
    cv2.createTrackbar('Brightness', winName, 255,255*2, Trackchanged)
    cv2.createTrackbar('Contrast', winName, 10,100, Trackchanged)
    cv2.createTrackbar('clipLimit', winName, 0,100, Trackchanged)
    cv2.createTrackbar('gridSize', winName, 1,100, Trackchanged)

    while True:
        # Cập nhật dữ liệu từ các thanh trượt
        randBG = cv2.getTrackbarPos('randBG', winName)
        gamma = cv2.getTrackbarPos('Gamma', winName)
        ksize = cv2.getTrackbarPos('Blur', winName)
        shift_x = cv2.getTrackbarPos('Shift_x', winName)
        shift_y = cv2.getTrackbarPos('Shift_y', winName)
        scale = cv2.getTrackbarPos('Scale', winName)/10
        angle = 180 - cv2.getTrackbarPos('Rotate', winName)
        alpha = cv2.getTrackbarPos('Contrast', winName)/10
        beta = cv2.getTrackbarPos('Brightness', winName) - 255
        clipLimit = cv2.getTrackbarPos('clipLimit', winName)
        gridSize = cv2.getTrackbarPos('gridSize', winName)

        # Các phép không làm thay đổi tọa độ bbox

        # Thay đổi background
        bboxes = open(bbox_dir, 'r').read().splitlines()
        if len(bboxes) > 1:
            sys.exit('Chon anh co 1 vat thoi')
        bbox = [float(i) for i in bboxes[0].split(' ')[1:]]
        x_center = int(bbox[0] * width)
        y_center = int(bbox[1] * height)
        tl_bbx = [int((bbox[0] - bbox[2] / 2) * width), int((bbox[1] - bbox[3] / 2) * height)]
        br_bbx = [int((bbox[0] + bbox[2] / 2) * width), int((bbox[1] + bbox[3] / 2) * height)]
        mask = cv2.imread(mask_dir,0)
        mask[0:tl_bbx[1] + 1, :] = 0
        mask[br_bbx[1]:height, :] = 0

        if gridSize:
            clahe_mat = cv2.createCLAHE(clipLimit, 2*(gridSize, ))
            img_lab = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2LAB)
            img_lab[:, :, 0] = clahe_mat.apply(img_lab[:, :, 0])
            clahe = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

        if randBG:
            mask_bgr = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            bg_rand = (np.random.rand(height,width,channels)*256).astype(np.uint8)
            bg_rand = cv2.bitwise_and(bg_rand,255-mask_bgr)
            fg = cv2.bitwise_and(clahe,mask_bgr)
            bgChanged = cv2.add(fg, bg_rand)

        # Làm mờ
        if ksize % 2:
            if randBG:
                blurred = cv2.GaussianBlur(bgChanged, (ksize, ksize), 0)
            else:
                blurred = cv2.GaussianBlur(img.copy(), (ksize, ksize), 0)

            # blurred_gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
            # print(cv2.Laplacian(blurred_gray,cv2.CV_64F).var())
        else:
            cv2.waitKey(1)
            continue

        # Thay đổi gamma
        if gamma:
            invGamma = 30 / gamma # Đúng ra là 1/gamma nhưng làm thế để chạy được số thập phân (có thể làm tối)
            table = np.array([((i / 255.0) ** invGamma) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
            gammaCorrected = cv2.LUT(blurred, table)
        else:
            cv2.waitKey(1)
            continue

        # Thay đổi độ sáng và độ tương phản
        if alpha:
            table = np.array([])
            for i in np.arange(0,256):
                new_value = i*alpha + beta
                if new_value < 0:
                    new_value = 0
                if new_value > 255:
                    new_value = 255
                table = np.append(table,new_value)
            table = table.astype('uint8')
            brightness_and_contrast_changed = cv2.LUT(gammaCorrected, table)
        else:
            cv2.waitKey(1)
            continue


        # Các phép làm thay đổi tọa độ bbox

        # Xoay
        M = cv2.getRotationMatrix2D((x_center, y_center), angle, 1.0) # Lấy ma trận xoay
        rotated = cv2.warpAffine(brightness_and_contrast_changed, M, (width, height))   # Xoay ảnh
        rotated_mask = cv2.warpAffine(mask, M, (width, height)) # Xoay mask
        # y_coords_white_pixel,x_coords_white_pixel = np.where(rotated_mask == 255)
        # tl_mask = (min(x_coords_white_pixel),min(y_coords_white_pixel))
        # br_mask = (max(x_coords_white_pixel),max(y_coords_white_pixel))
        bbx_mask = cv2.boundingRect(rotated_mask)   # Tính lại bbox sau khi xoay
        tl_mask = [bbx_mask[0], bbx_mask[1]] # Top left mới
        br_mask = [bbx_mask[0] + bbx_mask[2] - 1, bbx_mask[1] + bbx_mask[3] - 1]    # Bottom right mới
        # cv2.rectangle(rotated_mask, tl_mask, br_mask, 255)
        # cv2.imshow('mask',rotated_mask)

        # Dịch ảnh
        M = np.float32([[1, 0, shift_x-width], [0, 1, shift_y-height]])
        shifted = cv2.warpAffine(rotated, M, (width, height),borderMode=cv2.BORDER_CONSTANT)

        tl_mask[0] += shift_x-width
        tl_mask[1] += shift_y-height
        br_mask[0] += shift_x-width
        br_mask[1] += shift_y-height

        # Resize
        try:
            resized = cv2.resize(shifted,(int(width*scale),int(height*scale)))
            resized1 = cv2.resize(resized,(width,height))

            tl_mask[0] = int(tl_mask[0]*scale)
            tl_mask[1] = int(tl_mask[1]*scale)
            br_mask[0] = int(br_mask[0]*scale)
            br_mask[1] = int(br_mask[1]*scale)

        except:
            cv2.waitKey(1)
            continue


        final = resized
        # Hiển thị kết quả nhận dạng
        results = model.detect(final)
        for result in results:
            cv2.putText(final, result['name'] + ' ' + str(result['confidence']) + '%',
                        (result['top left'][0], result['top left'][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.rectangle(final, result['top left'], result['bottom right'], (0, 0, 0), 2)
        # Hiển thị kết quả bbox người đánh
        # cv2.rectangle(final, tuple(tl_mask), tuple(br_mask), (255, 255, 0))
        cv2.imshow('Result image',final)
        cv2.waitKey(1)


def get_mask(removed_bg_folder_dir, save_folder_dir, org_folder_dir):
    removed_bg_filename = [i for i in os.listdir(removed_bg_folder_dir)]
    index = 2
    angle = 0
    toggle = None
    while index < len(removed_bg_filename):
        # Tạo mask
        removed_bg_dir = os.path.join(removed_bg_folder_dir, removed_bg_filename[index])
        removed_bg = cv2.imread(removed_bg_dir)
        removed_bg_hsv = cv2.cvtColor(removed_bg, cv2.COLOR_BGR2HSV)
        mask = 255 - cv2.inRange(removed_bg_hsv, (103, 220, 243), (103, 220, 243))
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Lấy kích thước ảnh gốc để resize mask lại cho bằng ảnh gốc
        org_filename = removed_bg_filename[index].replace('-removebg-preview', '')
        org_dir = os.path.join(org_folder_dir, org_filename)
        if not os.path.exists(org_dir): # Đổi lại đuôi nếu ảnh gốc không phải .png
            org_dir = org_dir.replace('png', 'jpg')
        org = cv2.imread(org_dir)
        h, w = org.shape[:2]
        mask_resized = cv2.resize(mask_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

        # Lưu mask
        mask_filename = removed_bg_filename[index].replace('-removebg-preview', '_mask')
        mask_dir = os.path.join(save_folder_dir, mask_filename)
        cv2.imwrite(mask_dir, mask_resized)

        # Test mask
        mask_check_bgr = cv2.imread(mask_dir)
        # if  np.all((mask_check_bgr > 0) & (mask_check_bgr < 255)):
        #     print(mask_filename + ': co pixel khac 0 va 255')
        # else:
        #     print(mask_filename + ': tat ca pixel deu ok')
        mask_check_gray = cv2.cvtColor(mask_check_bgr, cv2.COLOR_BGR2GRAY)
        bbox = cv2.boundingRect(mask_check_gray)
        x_center = bbox[0] + bbox[2] / 2 - 1
        y_center = bbox[1] + bbox[3] / 2 - 1
        rotate_matrix = cv2.getRotationMatrix2D((x_center, y_center), angle, 1.0)  # Lấy ma trận xoay
        org = cv2.warpAffine(org, rotate_matrix, (w, h))  # Xoay ảnh
        mask_check_bgr = cv2.warpAffine(mask_check_bgr, rotate_matrix, (w, h))  # Xoay mask
        mask_check_gray = cv2.cvtColor(mask_check_bgr, cv2.COLOR_BGR2GRAY)
        bbox = cv2.boundingRect(mask_check_gray)  # Tính lại bbox sau khi xoay
        top_left = tuple(bbox[0:2])
        bottom_right = (bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1)

        cv2.rectangle(mask_check_bgr, top_left, bottom_right, (255, 0, 255), 1)
        cv2.rectangle(org, top_left, bottom_right, (255, 0, 255), 1)
        info_panel = np.zeros((98, 640*2, 3), np.uint8)
        info_panel[:,:] = [50, 50, 50]
        info = 'Current mask: ' + mask_filename + ' -- Index: ' + str(index)
        cv2.putText(info_panel, info, (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 1, lineType=cv2.LINE_AA)
        check_img = cv2.vconcat([info_panel, cv2.hconcat([org, mask_check_bgr])])
        cv2.imshow('Check', check_img)
        toggle = cv2.waitKeyEx()
        if toggle == 2555904:
            index += 1
            angle = 0
        elif toggle == 2424832:
            index -= 1
            angle = 0
        elif toggle == 45:
            angle += 2
        elif toggle == 43:
            angle -= 2



def testLineError(video_dir=None, cam=None):
    if video_dir:
        cap = cv2.VideoCapture(video_dir)
    else:
        cap = cv2.VideoCapture(cam)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    def callback(x):
        pass

    winName = 'Test line'
    cv2.namedWindow(winName,0)
    if video_dir:
        numFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.createTrackbar('idxFrame',winName,1,numFrame, callback)
    cv2.createTrackbar('roi_tlx', winName, 199, w, callback)
    cv2.createTrackbar('roi_tly', winName, 321, h, callback)
    cv2.createTrackbar('roi_brx', winName, 428, w, callback)
    cv2.createTrackbar('roi_bry', winName, 434, h, callback)
    cv2.createTrackbar('thksize', winName, 37, 50, callback)
    cv2.createTrackbar('c', winName, 0, 20, callback)
    cv2.createTrackbar('bksize', winName, 7, 50, callback)
    cv2.createTrackbar('maxThres', winName, 0, 255, callback)
    cv2.createTrackbar('minThres', winName, 0, 255, callback)
    cv2.createTrackbar('linThres', winName, 90, 200, callback)
    cv2.createTrackbar('minLinLen',winName,74,100,callback)
    cv2.createTrackbar('maxLinGap',winName,5,50,callback)

    while 1:
        roi_tlx = cv2.getTrackbarPos('roi_tlx',winName)
        roi_tly = cv2.getTrackbarPos('roi_tly',winName)
        roi_brx = cv2.getTrackbarPos('roi_brx',winName)
        roi_bry = cv2.getTrackbarPos('roi_bry',winName)
        thksize = cv2.getTrackbarPos('thksize',winName)
        if not thksize % 2 or not thksize > 1:
            cv2.waitKey(1)
            continue
        c = cv2.getTrackbarPos('c',winName)
        bksize = cv2.getTrackbarPos('bksize',winName)
        if not bksize % 2:
            cv2.waitKey(1)
            continue
        maxThres = cv2.getTrackbarPos('maxThres',winName)
        minThres = cv2.getTrackbarPos('minThres',winName)
        linThres = cv2.getTrackbarPos('linThres',winName)
        minLinLen = cv2.getTrackbarPos('minLinLen',winName)
        maxLinGap = cv2.getTrackbarPos('maxLinGap',winName)
        if video_dir:
            idxFrame = cv2.getTrackbarPos('idxFrame', winName)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idxFrame - 1)

        frame = cap.read()[1]
        if frame is None:
            sys.exit('Cannot grab the frame')
        frame_roi = frame[roi_tly:roi_bry, roi_tlx:roi_brx].copy()
        if not frame_roi.any():
            cv2.waitKey(1)
            continue
        roi_gray = cv2.cvtColor(frame_roi,cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(roi_gray,(bksize,bksize),0)
        thresholded = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,thksize,c)
        cannied = cv2.Canny(blurred,minThres,maxThres)
        # laplacian = cv2.Laplacian(roi_gray,)
        cv2.imshow('before_opening',thresholded)
        kernel1 = np.array([[0,0,1],[0,1,0], [1,0,0]]).astype(np.uint8)
        kernel2 = np.array([[0,0,0,1,0],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0], [0,1,0,0,0]]).astype(np.uint8)
        thresholded = cv2.morphologyEx(thresholded,cv2.MORPH_OPEN,kernel2)
        cv2.imshow('after_opening',thresholded)
        lines = cv2.HoughLinesP(thresholded.copy(),1,np.pi/180,linThres,minLineLength=minLinLen,maxLineGap=maxLinGap)
        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         rho = lines[i][0][0]
        #         theta = lines[i][0][1]
        #         a = math.cos(theta)
        #         b = math.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        #         cv2.line(frame, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(frame, (roi_tlx+l[0], roi_tly+l[1]), (roi_tlx+l[2], roi_tly+l[3]), (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.imshow('Line test',thresholded)
        cv2.imshow('frame',frame)
        # cv2.imshow('cannied',cannied)
        cv2.waitKey(1)


class Data:
    def __init__(self,img_folder_dir=None):
        self.img_folder_dir = img_folder_dir

    def test_label(self, label_folder_dir):
        for label_filename in label_folder_dir:
            label_dir = os.path.join(label_folder_dir,label_filename)
            label = open(label_dir,'r').readlines()
            label = [i.split('\n')[0].split(' ') for i in label]

            # Đang viết dở

            img_name = label_filename.split('.')[0]
            img_dir = os.path.join(self.img_folder_dir, img_name + '.png')
            if not os.path.exists(img_dir):
                img_dir = os.path.join(self.img_folder_dir, img_name + '.jpg')
            img = cv2.imread(img_dir)

    def augment(self, sequence, img_save_folder_dir=None, label_save_folder_dir=None,
                mask_folder_dir=None,label_folder_dir=None, num_img=None):

        # Check xem có thiếu mask trong trường hợp rotate không
        if ('rotate' in [i[0] for i in sequence] \
                or 'background' in [i[0] for i in sequence]) \
                and not mask_folder_dir:
            sys.exit('Them mask_folder_dir neu muon xoay anh chinh xac')

        # Lấy image dictionary
        img_dict = {}
        shape_dict = {}
        mask_dict = {}
        for img_filename in os.listdir(self.img_folder_dir):
            img_name = img_filename.split('.')[0]
            img = cv2.imread(os.path.join(self.img_folder_dir, img_filename))
            img_dict[img_name] = img
            shape_dict[img_name] = img.shape[:2]

        # Lấy bounding box dictionary
        bbox_dict = {}
        mask_dict = {}
        if mask_folder_dir:
            for mask_filename in os.listdir(mask_folder_dir):
                img_name = mask_filename.split('_mask')[0]
                mask = cv2.imread(os.path.join(mask_folder_dir, mask_filename), 0)
                bbox_dict[img_name] = list(cv2.boundingRect(mask))
                mask_dict[img_name] = mask
        elif label_folder_dir:
            for label_filename in os.listdir(label_folder_dir):
                if label_filename == 'classes.txt':
                    continue
                label_filepath = os.path.join(label_folder_dir, label_filename)
                label_file = open(label_filepath, 'r')
                bbox = label_file.readlines()[0].split('\n')[0].split(' ')
                bbox = [float(i) for i in bbox[1:]]
                img_name = label_filename.split('.txt')[0]
                h, w = shape_dict[img_name]
                bbox[0] = round((bbox[0] - bbox[2] / 2) * w)
                bbox[1] = round((bbox[1] - bbox[3] / 2) * h)
                bbox[2] = round(bbox[2] * w)
                bbox[3] = round(bbox[3] * h)
                bbox_dict[img_name] = bbox
                label_file.close()

                # Test bbox sau khi lấy từ data
                # img = img_dict[img_name]
                # top_left = tuple(bbox_dict[img_name][:2])
                # bottom_right = (bbox_dict[img_name][0] + bbox_dict[img_name][2] - 1,
                #                 bbox_dict[img_name][1] + bbox_dict[img_name][3] - 1)
                # cv2.rectangle(img, top_left, bottom_right, (0, 255, 0))
                # cv2.imshow(img_name, img)
                # cv2.waitKey()
                # cv2.destroyAllWindows()

        else:
            sys.exit('Them mask_folder_dir hoac label_folder_dir')

        # Khởi tạo các transformer của albumentations
        motion_blur = a.Compose([a.MotionBlur(blur_limit=(param[1], param[2]), p=1)])
        clahe = a.Compose([a.CLAHE])

        # Làm dày
        num_img_dict = {i: 0 for i in img_dict}
        num_img_total = 0
        while True:
            for img_name in img_dict:
                img = img_dict[img_name].copy()
                bbox = bbox_dict[img_name].copy()
                if mask_folder_dir:
                    mask = mask_dict[img_name].copy()
                for param in sequence:
                    # param = [tên phương pháp biến đổi ảnh (shift, rotate,...),
                    #          phạm vi min (= -1 nếu không cần),
                    #          phạm vi max (= -1 nếu không cần),
                    #         % xác suất xảy ra (max = 100)]
                    if random.random() * 100 < param[3]:
                        h_img, w_img, deep = img.shape

                        if param[0] == 'gaussian blur':
                            ksize = random.randrange(param[1], param[2] + 1, 2)
                            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

                        elif param[0] == 'motion blur':
                            img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
                            transformed = motion_blur(image=img_rgb)['image']
                            img = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)


                        elif param[0] == 'gamma':
                            gamma = random.uniform(param[1], param[2])
                            inv_gamma = 1 / gamma
                            table = np.array([((i / 255.0) ** inv_gamma) * 255
                                              for i in np.arange(0, 256)]).astype("uint8")
                            img = cv2.LUT(img, table)

                        elif param[0] == 'background':
                            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                            img_rand = (np.random.rand(h_img, w_img, deep) * 256).astype(np.uint8)
                            bg_rand = cv2.bitwise_and(img_rand, 255 - mask_bgr)
                            fg = cv2.bitwise_and(img.copy(), mask_bgr)
                            img = cv2.add(fg, bg_rand)

                        elif param[0] == 'rotate':
                            x_center = bbox[0] + bbox[2] / 2 - 1
                            y_center = bbox[1] + bbox[3] / 2 - 1

                            # Giới hạn góc xoay để không bị mất vật
                            w_half_cross = x_center - bbox[0] + 1
                            h_half_cross = y_center - bbox[1] + 1
                            len_half_cross = math.sqrt(w_half_cross**2 + h_half_cross**2)
                            dis_to_v_axis = min(x_center + 1, w_img - x_center)
                            dis_to_h_axis = min(y_center + 1, h_img - y_center)
                            min_dis_to_axis = min(dis_to_v_axis, dis_to_h_axis)
                            if min_dis_to_axis < len_half_cross:
                                if dis_to_h_axis < dis_to_v_axis:
                                    angle_cross_axis = math.acos(bbox[3]/2/len_half_cross)
                                else:
                                    angle_cross_axis = math.acos(bbox[2]/2/len_half_cross)
                                angle_touchpoint_axis = math.acos(min_dis_to_axis/len_half_cross)
                                max_angle = math.degrees(angle_cross_axis - angle_touchpoint_axis)
                                max_angle = min(param[2], max_angle)
                            else:
                                max_angle = param[2]
                            if random.random() * 100 < param[3]:
                                angle = random.uniform(-max_angle, max_angle)
                                rotate_matrix = cv2.getRotationMatrix2D((x_center, y_center), angle, 1.0)
                                img = cv2.warpAffine(img, rotate_matrix, (w_img, h_img))
                                mask = cv2.warpAffine(mask, rotate_matrix, (w_img, h_img))  # Xoay ảnh
                                bbox = list(cv2.boundingRect(mask))  # Tính lại bbox sau khi xoay
                                # Test bbox sau khi xoay
                                # tl_mask = [bbox[0], bbox[1]]  # Top left mới
                                # br_mask = [bbox[0] + bbox[2] - 1,
                                #            bbox[1] + bbox[3] - 1]  # Bottom right mới
                                # cv2.rectangle(mask, tl_mask, br_mask, 255)
                                # cv2.imshow('mask', mask)
                        elif param[0] == 'shift':
                            top_left_x, top_left_y = bbox[0], bbox[1]
                            w_bbox, h_bbox = bbox[2], bbox[3]
                            min_tx, min_ty = - top_left_x, - top_left_y
                            max_tx = w_img - (top_left_x + w_bbox)
                            max_ty = h_img - (top_left_y + h_bbox)
                            tx = random.randint(min_tx, max_tx)
                            ty = random.randint(min_ty, max_ty)
                            shift_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                            img = cv2.warpAffine(img, shift_matrix, (w_img, h_img), borderMode=cv2.BORDER_CONSTANT)
                            if 'rotate' in [param[0] for param in sequence]:
                                mask = cv2.warpAffine(mask, shift_matrix, (w_img, h_img), borderMode=cv2.BORDER_CONSTANT)
                            bbox[0] += tx
                            bbox[1] += ty

                # Lưu ảnh mới tạo
                num_img_dict[img_name] += 1
                img_save_name = img_name + '_aug_' + str(num_img_dict[img_name]) + '.png'
                img_save_dir = os.path.join(img_save_folder_dir, img_save_name)
                cv2.imwrite(img_save_dir, img)

                # Test bbox sau khi làm dày
                top_left = (bbox[0], bbox[1])
                bottom_right = (bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1)
                cv2.rectangle(img, top_left, bottom_right, (0, 255, 0))
                cv2.imshow(img_name, img)
                cv2.waitKey()
                cv2.destroyAllWindows()

                # Lưu label cho ảnh mới tạo
                label_param = [None]*5
                obj_name = img_name.split('_')[0]
                if obj_name == 'ball':
                    label_param[0] = 0
                elif obj_name == 'damper':
                    label_param[0] = 1
                elif obj_name == 'clamp':
                    label_param[0] = 2
                label_param[1] = (bbox[0] + bbox[2]/2 - 1) / w_img
                label_param[2] = (bbox[1] + bbox[3]/2 - 1) / h_img
                label_param[3] = bbox[2] / w_img
                label_param[4] = bbox[3] / h_img
                label_dir = os.path.join(label_save_folder_dir, img_save_name.replace('.png', '.txt'))
                label_file = open(label_dir,'w')
                label_file.write(str(label_param).replace('[', '')
                                 .replace(', ', ' ').replace(']', '\n'))
                label_file.close()

                num_img_total += 1
                print(str(num_img_total) + '/' + str(num_img))
                if num_img_total >= num_img:
                    # print(num_img_dict)
                    return


get_mask(removed_bg_folder_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Images\Removedbg',
         save_folder_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Images\Mask',
         org_folder_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Images\Gamma_corrected')

# data = Data(img_folder_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Images\Raw')
# data.augment(sequence=[['background', -1, -1, 100],
#                        ['gauss blur', 3, 21, 100],
#                        ['gamma', 1, 3, 100],
#                        ['shift', -1, -1, 100],
#                        ['rotate', -45, 45, 100]],
#              img_save_folder_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Images\Augmented',
#              label_save_folder_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Labels\Augmented',
#              mask_folder_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Images\Mask',
#              num_img=222)
# data.augment(sequence=[['shift', -1, -1, 100], ['rottate', -45, 45, 100]],
#              img_save_folder_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Images\Augmented',
#              label_save_folder_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Labels\Augmented',
#              label_folder_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Labels\Origin',
#              num_img=222)


# testLineError(video_dir=r'D:\WON\DO_AN\Data\Line\line_2.avi')

# testSegmentColor(r'D:\WON\DO_AN\Data\Training\Lan1\Damper\Images\Foreground\Raw\damper_20_1_segmented.jpg')

# modelTest(img_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Images\Raw\clamp_100_1.jpg',
#           bbox_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Labels\Origin\clamp_100_1.txt',
#           mask_dir=r'D:\WON\DO_AN\Data\Training\Lan1\Clamp\Images\Mask\clamp_100_1_mask.png')

# cv2.imshow("result",gamma_correct(img_dir='D:\\WON\\DO_AN\\Changed_data\\49.jpg',gamma=2))
# cv2.waitKey(0)

# getdata4linReg(data_dir='D:\\WON\\DO_AN\\Data\\Distance_estimate',
#                weights_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3_best.weights',
#                cfg_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\yolov3.cfg',
#                names_dir='D:\\WON\\DO_AN\\Code\\Model\\YOLOv3\\obj.names',
#                save_dir='D:\\WON\\DO_AN\\Code\\Quan')
# time.sleep(1)

# linear_regression(inv_w_array_dir='D:\\WON\\DO_AN\\Code\\Quan\\inv_w_array.npy',
#                   dis_array_dir='D:\\WON\\DO_AN\\Code\\Quan\\dis_array.npy')
# print(butter_lowpass_filter(data=[0,10],cutoff=1,fs=10,order=5))

# for image in os.listdir('D:\\WON\\DO_AN\\Cho_thay_xem'):
#     frame = cv2.imread('D:\\WON\\DO_AN\\Cho_thay_xem\\'+image)
#     detected_frame = yolov3_detect(frame)
#     cv2.imshow(image,detected_frame)
#     cv2.waitKey(0)
#     cv2.imwrite('D:\\WON\\DO_AN\\Cho_thay_xem\\'+image,detected_frame)

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


