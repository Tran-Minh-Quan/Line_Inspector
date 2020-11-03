import cv2
import numpy as np
import imutils
import math
from scipy.signal import butter, filtfilt


class DistanceMultiClasses:
    def __init__(self, focal_length):
        self.focal_length = focal_length
        self.damper = CircleDistance(0, 50000, 1, 30, 867.7887424687945, -0.18242145320198588, 764)
        self.ball = CircleDistance(0, 50000, 1, 50, 7269.07267232745*2, -7.5821004063644395, 764)
        self.clamp = EdgeDistance(0, 500, 7160.170584232902, -7.36958918477265, 764)

    def calculate(self, img, top_left, bot_right, extended_ratio, object_name, mode):
        if object_name is "damper":
            return self.damper.low_pass_calc(img, top_left, bot_right, extended_ratio, mode)
        elif object_name is "ball":
            return self.ball.low_pass_calc(img, top_left, bot_right, extended_ratio, mode)
        elif object_name is "clamp":
            return self.clamp.edge_based_calculate(img, top_left, bot_right, extended_ratio, mode)


class CircleDistance:
    def __init__(self, low_canny, high_canny, step_size, hough_param, slope, intercept, focal_length):
        self.NO_ERROR = 0
        self.INVALID_INPUT_ERROR = 1
        self.NON_CIRCLE_ERROR = 2   # Circle undetectable
        self.MULTIPLE_CIRCLES_ERROR = 3
        self.low_canny = low_canny
        self.high_canny = high_canny
        self.step_size = step_size
        self.slope = slope  # Slope of linear regression distance estimate function
        self.intercept = intercept  # Intercept of linear regression distance estimate function
        self.hough_param = hough_param  # This value should varies between 20-70 (30 is best for some cases)
        self.focal_length = focal_length
        self.first_detect = 1
        self.redetect = 1
        self.last_canny_param = 450
        self.distance_array = np.zeros(16)
        self.distance_filtered = 0
        self.normal_cutoff = 1/6
        self.order = 4
        [self.b, self.a] = butter(self.order, self.normal_cutoff, btype='low', analog=False, output='ba')

    def calculate(self, img, top_left, bot_right, extended_ratio, mode):
        try:
            img_out = img.copy()
        except Exception:
            self.redetect = 1
            return [-1, np.zeros([500, 500]), self.INVALID_INPUT_ERROR]
        width_box = bot_right[1] - top_left[1]
        height_box = bot_right[0] - top_left[0]
        # Check for invalid input
        if width_box <= 0 or height_box <= 0 \
                or max(bot_right[0], top_left[0]) > img.shape[0] or max(bot_right[1], top_left[1]) > img.shape[1]:
            return [-1, img_out, self.INVALID_INPUT_ERROR]
        # mode = 0 when not using circle detection
        # mode = 1 when using circle detection (default)
        if mode == 0:
            distance = calculate_distance_linear(self.slope, self.intercept, width_box)
            cv2.rectangle(img_out, tuple(top_left), tuple(bot_right), color=(255, 0, 0), thickness=2)
            return [distance, img_out, 0]
        elif mode != 1:
            print("Wrong mode selected")
            return [-1, img_out, self.INVALID_INPUT_ERROR]
        else:
            pass
        # Calculate x axis extended
        x_axis_extended = [max(0, int(top_left[0] - extended_ratio * width_box)),
                           min(img.shape[0], int(bot_right[0] + extended_ratio * width_box))]
        # Calculate y axis extended
        y_axis_extended = [max(0, int(top_left[1] - extended_ratio * height_box)),
                           min(img.shape[1], int(bot_right[1] + extended_ratio * height_box))]
        # Crop image
        crop_img = img[x_axis_extended[0]: x_axis_extended[1], y_axis_extended[0]:y_axis_extended[1]]
        cv2.rectangle(img_out, (y_axis_extended[0], x_axis_extended[0]), (y_axis_extended[1], x_axis_extended[1]),
                      (255, 0, 0), 2)
        # cv2.imwrite("img_test.jpg", crop_img)
        # Grayscale image
        if crop_img.ndim > 2:
            gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)   # Convert for color image
        else:
            gray_img = crop_img     # Gray scale image
        # Binary search algorithm to find RIGHTMOST Canny parameter
        if self.redetect == 1:
            rm_left = self.low_canny
            rm_right = self.high_canny
        else:
            rm_left = max(self.last_canny_param - 2000, 0)
            rm_right = self.last_canny_param + 2000
        canny_param = rightmost_canny_param_search(gray_img, rm_left, rm_right,
                                                   self.step_size, self.hough_param)
        # Detect circle with determined Canny parameter
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=canny_param, param2=self.hough_param, minRadius=0, maxRadius=0)
        # Return error if circle is undetectable
        if circles is None or circles[0][0][2] == 0:
            self.redetect = 1
            distance = calculate_distance_linear(self.slope, self.intercept, width_box)
            return [distance, img_out, self.NON_CIRCLE_ERROR]
        circles_round = np.round(circles[0, :]).astype("int")
        # Mark detected circle in image
        for (y, x, r) in circles_round:
            cv2.circle(img_out, (y + y_axis_extended[0], x + x_axis_extended[0]), r, (0, 255, 0), 4)
            cv2.rectangle(img_out, (y - 5 + y_axis_extended[0], x - 5 + x_axis_extended[0]),
                          (y + 5 + y_axis_extended[0], x + 5 + x_axis_extended[0]), (0, 128, 255), -1)
        if circles.shape[1] > 1:
            self.redetect = 1
            distance = calculate_distance_linear(self.slope, self.intercept, width_box)
            return [distance, img_out, self.MULTIPLE_CIRCLES_ERROR]
        radius_pixel = circles[0][0][2]
        # Calculate distance with linear regression function
        # print(radius_pixel)  # Delete after testing
        distance = calculate_distance_linear(self.slope, self.intercept, 2*radius_pixel)
        # distance = self.slope * 1/radius_pixel + self.intercept
        self.last_canny_param = canny_param
        self.redetect = 0
        return [distance, img_out, self.NO_ERROR]

    def low_pass_calc(self, img, top_left, bot_right, extended_ratio, mode):
        distance, img_out, error_code = self.calculate(img, top_left, bot_right, extended_ratio, mode)
        if error_code == 0:
            if self.first_detect == 1:
                self.distance_array = [distance]*16
                self.distance_filtered = distance
                self.first_detect = 0
            else:
                for j in range(0, len(self.distance_array) - 1):
                    self.distance_array[j] = self.distance_array[j + 1]
                self.distance_array[len(self.distance_array) - 1] = distance
                self.distance_filtered = filtfilt(self.b, self.a, self.distance_array)[15]
        else:
            pass
        return [self.distance_filtered, img_out, error_code]


class EdgeDistance:
    def __init__(self, low_canny, high_canny, slope, intercept, focal_length):
        self.low_canny = low_canny
        self.high_canny = high_canny
        self.slope = slope
        self.intercept = intercept
        self.focal_length = focal_length

    def edge_based_calculate(self, img, top_left, bot_right, extended_ratio, mode):
        try:
            img_out = img.copy()
        except Exception:
            return [-1, np.zeros([500, 500]), 1]
        width_box = bot_right[1] - top_left[1]
        height_box = bot_right[0] - top_left[0]
        # Check for invalid input
        if width_box <= 0 or height_box <= 0 \
                or max(bot_right[0], top_left[0]) > img.shape[0] or max(bot_right[1], top_left[1]) > img.shape[1]:
            return [-1, img_out, 1]
        cv2.rectangle(img_out, tuple(top_left), tuple(bot_right), color=(255, 0, 0), thickness=2)
        if mode == 0:
            distance = calculate_distance_linear(self.slope, self.intercept, width_box)
            return [distance, img_out, 0]
        elif mode != 1:
            print("Wrong mode selected")
            return [-1, img_out, 1]
        else:
            pass
        kernel = np.ones((5, 5), np.uint8)  # Init kernel for erosion operation
        width_box = bot_right[1] - top_left[1]  # Calculate width of bounding box
        height_box = bot_right[0] - top_left[0]  # Calculate height of bounding box
        if width_box == 0 or height_box == 0:
            return [-1, img_out, 1]
        # Calculate x axis extended
        x_axis_extended = [max(0, int(top_left[1] - extended_ratio * width_box)),
                           min(img.shape[0], int(bot_right[1] + extended_ratio * width_box))]
        # Calculate y axis extended
        y_axis_extended = [max(0, int(top_left[0] - extended_ratio * height_box)),
                           min(img.shape[1], int(bot_right[0] + extended_ratio * height_box))]
        # Crop image
        crop_img = img[x_axis_extended[0]: x_axis_extended[1], y_axis_extended[0]:y_axis_extended[1]]
        erode = cv2.erode(crop_img, kernel, iterations=1)  # Perform erosion operation on input image
        dilate = cv2.dilate(erode.copy(), kernel, iterations=1)
        edges = cv2.Canny(dilate, self.low_canny, self.high_canny)
        contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours of the image
        contours = imutils.grab_contours(contours)  # Grabs the appropriate tuple value
        c = sorted(contours, key=cv2.contourArea)  # Sort contours based on contour area
        ext_left, ext_right, ext_top, ext_bot = 1e5, -1e5, 1e5, -1e5  # Initialize external boundary on contours
        # Find external boundary of contours
        if len(c) > 0:
            for cc in c:
                ext_left = min(cc[:, :, 0].min(), ext_left)
                ext_right = max(cc[:, :, 0].max(), ext_right)
                ext_top = min(cc[:, :, 1].min(), ext_top)
                ext_bot = max(cc[:, :, 1].max(), ext_bot)
            try:
                if ext_right - ext_left == 0:
                    return [-1, img, 1]
                # distance = calculate_distance(object_width, focal_length, ext_right - ext_left)  # Calculate distance
                distance = calculate_distance_linear(self.slope, self.intercept, ext_right - ext_left)
                # Draw contour for further troubleshooting
                # img_1 = cv2.drawContours(crop_img.copy(), contours, -1, (0, 0, 255), 3)
                for contour in contours:
                    contour[:, :, 0] += y_axis_extended[0]
                    contour[:, :, 1] += x_axis_extended[0]
                img_out = cv2.drawContours(img_out, contours, -1, (0, 0, 255), 3)
                # Draw rectangle used for distance estimation
                # img_out = img_1.copy()
                cv2.rectangle(img_out, (ext_left+y_axis_extended[0], ext_top+x_axis_extended[0]),
                              (ext_right+y_axis_extended[0],
                               ext_bot+x_axis_extended[0]), (0, 255, 0), 3)
                error = 0
            except Exception:
                error = 1
                distance = -1
        else:
            distance = calculate_distance_linear(self.slope, self.intercept, width_box)
            return [distance, img_out, 1]
        return [distance, img_out, error]


def calculate_distance_linear(slope, intercept, width_pixel):
    return slope * 1/width_pixel + intercept


def rightmost_canny_param_search(gray_img, rm_left, rm_right, step_size, hough_param):
    step = 0
    while rm_left < rm_right:
        step += 1
        canny_param = math.floor((rm_right + rm_left) / (2 * step_size)) * step_size
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=canny_param, param2=hough_param, minRadius=0, maxRadius=0)
        if circles is None or circles[0][0][2] == 0:
            rm_right = canny_param
        else:
            rm_left = canny_param + step_size
    # print("Iteration step taken: %d" % step)
    return rm_left - step_size


"Old references start"


def edge_based_calc_old(img, start_point, end_point, extended_ratio, low_canny, high_canny, focal_length, object_width):
    kernel = np.ones((5, 5), np.uint8)  # Init kernel for erosion operation
    width_box = end_point[1] - start_point[1]   # Calculate width of bounding box
    height_box = end_point[0] - start_point[0]  # Calculate height of bounding box
    if width_box == 0 or height_box == 0:
        return [-1, img, 1]
    # Calculate x axis extended
    x_axis_extended = [max(0, int(start_point[1] - extended_ratio * width_box)),
                       min(img.shape[0], int(end_point[1] + extended_ratio * width_box))]
    # Calculate y axis extended
    y_axis_extended = [max(0, int(start_point[0] - extended_ratio * height_box)),
                       min(img.shape[1], int(end_point[0] + extended_ratio * height_box))]
    # Crop image
    crop_img = img[x_axis_extended[0]: x_axis_extended[1], y_axis_extended[0]:y_axis_extended[1]]
    erode = cv2.erode(crop_img, kernel, iterations=1)   # Perform erosion operation on input image
    dilate = cv2.dilate(erode.copy(), kernel, iterations=1)  # Perform dilation operation on input image
    edges = cv2.Canny(dilate, low_canny, high_canny)  # Perform Canny edge detection algorithm on eroded image
    contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   # Find contours of the image
    contours = imutils.grab_contours(contours)  # Grabs the appropriate tuple value
    c = sorted(contours, key=cv2.contourArea)   # Sort contours based on contour area
    ext_left, ext_right, ext_top, ext_bot = 1e5, -1e5, 1e5, -1e5    # Initialize external boundary on contours
    # Find external boundary of contours
    if len(c) > 0:
        for cc in c:
            ext_left = min(cc[:, :, 0].min(), ext_left)
            ext_right = max(cc[:, :, 0].max(), ext_right)
            ext_top = min(cc[:, :, 1].min(), ext_top)
            ext_bot = max(cc[:, :, 1].max(), ext_bot)
        try:
            if ext_right - ext_left == 0:
                return [-1, img, 1]
            distance = calculate_distance_old(object_width, focal_length, ext_right - ext_left)     # Calculate distance
            # Draw contour for further troubleshooting
            img_1 = cv2.drawContours(crop_img.copy(), contours, -1, (0, 0, 255), 3)
            # Draw rectangle used for distance estimation
            img_out = img_1.copy()
            cv2.rectangle(img_out, (ext_left, ext_top), (ext_right, ext_bot), (0, 255, 0), 3)
            error = 0
        except Exception:
            error = 1
            distance = -1
            img_out = img
    else:
        return [-1, crop_img, 1]
    return [distance, img_out, error]


def calculate_distance_old(object_width, focal_length, width_pixel):
    distance = int((10 * (focal_length * object_width) / width_pixel) + 0.5)
    return distance
