import cv2
import numpy as np
import imutils
import math


''' CircleDistance module:
    For initialize: 
    1. This module should be initialize before using, recommended params for Canny and Hough respectively 
    is (200, 456) and 30.
    2. Slope and intercept should be calculated in advanced.
    For using calculate function:
    1. Argument mode can be set to 0 (not using circle detection) or 1 (using circle detection), default is 1.
    2. Extended_ratio should varies between 0.2 - 0.5 for best result.
    About returning output:
    1. Distance will be set to -1 if an error occurred.
    2. Return image will highlight detected circle and its center.
    3. Error return can be 0,1,2 (NO_ERROR, INVALID_INPUT_ERROR, NON_CIRCLE_ERROR).
'''


class CircleDistance:
    def __init__(self, low_canny, high_canny, step_size, hough_param, slope, intercept):
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
        self.first_detect = 1
        self.last_canny_param = 450

    def calculate(self, img, top_left, bot_right, extended_ratio, **kwargs):
        img_out = img.copy()
        width_box = bot_right[1] - top_left[1]
        height_box = bot_right[0] - top_left[0]
        # Check for invalid input
        if width_box <= 0 or height_box <= 0 \
                or max(bot_right[0], top_left[0]) > img.shape[0] or max(bot_right[1], top_left[1]) > img.shape[1]:
            return [-1, img_out, self.INVALID_INPUT_ERROR]
        # mode = 0 when not using circle detection
        # mode = 1 when using circle detection (default)
        if "mode" in kwargs:
            if kwargs['mode'] == 0:
                if 'object_width' in kwargs:
                    distance = calculate_distance(kwargs['object_width'], 764, width_box)
                    cv2.rectangle(img_out, tuple(top_left), tuple(bot_right), color=(255, 0, 0), thickness=2)
                    return [distance, img_out, 0]
                else:
                    print("object_width is missing")
                    return [-1, img_out, self.INVALID_INPUT_ERROR]
            elif kwargs['mode'] != 1:
                print("Wrong mode selected")
                return [-1, img_out, self.INVALID_INPUT_ERROR]
        else:
            pass
        # Calculate x axis extended
        x_axis_extended = [max(0, int(top_left[1] - extended_ratio * width_box)),
                           min(img.shape[0], int(bot_right[1] + extended_ratio * width_box))]
        # Calculate y axis extended
        y_axis_extended = [max(0, int(top_left[0] - extended_ratio * height_box)),
                           min(img.shape[1], int(bot_right[0] + extended_ratio * height_box))]
        # Crop image
        crop_img = img[x_axis_extended[0]: x_axis_extended[1], y_axis_extended[0]:y_axis_extended[1]]
        # cv2.imwrite("img_test.jpg", crop_img)
        # Grayscale image
        gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
        # Binary search algorithm to find LEFTMOST Canny parameter
        if self.first_detect == 1:
            rm_left = self.low_canny
            rm_right = self.high_canny
            pass
        else:
            rm_left = self.last_canny_param - 100
            rm_right = self.last_canny_param + 100
        step = 0
        while rm_left < rm_right:
            step += 1
            canny_param = math.floor((rm_right + rm_left) / (2*self.step_size))*self.step_size
            circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100,
                                       param1=canny_param, param2=self.hough_param, minRadius=0, maxRadius=0)
            if circles is None:
                rm_right = canny_param
            else:
                rm_left = canny_param + self.step_size
        print("Iteration step taken: %d" % step)
        canny_param = rm_left - self.step_size
        # Detect circle with determined Canny parameter
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=canny_param, param2=self.hough_param, minRadius=0, maxRadius=0)
        # Return error if circle is undetectable
        if circles is None:
            self.first_detect = 1
            return [-1, img_out, self.NON_CIRCLE_ERROR]
        circles_round = np.round(circles[0, :]).astype("int")
        # Mark detected circle in image
        for (y, x, r) in circles_round:
            cv2.circle(img_out, (y + y_axis_extended[0], x + x_axis_extended[0]), r, (0, 255, 0), 4)
            cv2.rectangle(img_out, (y - 5 + y_axis_extended[0], x - 5 + x_axis_extended[0]),
                          (y + 5 + y_axis_extended[0], x + 5 + x_axis_extended[0]), (0, 128, 255), -1)
            cv2.rectangle(img_out, (y_axis_extended[0], x_axis_extended[0]), (y_axis_extended[1], x_axis_extended[1]),
                          (255, 0, 0), 2)
        if circles.shape[1] > 1:
            self.first_detect == 1
            return [-1, img_out, self.MULTIPLE_CIRCLES_ERROR]
        radius_pixel = circles[0][0][2]
        # Calculate distance with linear regression function
        distance = self.slope * 1/radius_pixel + self.intercept
        self.last_canny_param = canny_param
        self.first_detect = 0
        return [distance, img_out, self.NO_ERROR]


''' This is a simple distance estimate formula with known width object'''
def calculate_distance(object_width, focal_length, width_pixel):
    distance = int((10 * (focal_length * object_width) / width_pixel) + 0.5)
    return distance


''' edge_based module:
    1. This module calculate distance to object with known width, only works well when the background is flat.
    2. Error return can be 0,1 (NO_ERROR, ERROR_DETECTED).
    3. Focal length and object width should be measured in advanced.
'''
def edge_based(img, start_point, end_point, extended_ratio, canny_var_1, canny_var_2, focal_length, object_width):
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
    edges = cv2.Canny(dilate, canny_var_1, canny_var_2)  # Perform Canny edge detection algorithm on eroded image
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
            distance = calculate_distance(object_width, focal_length, ext_right - ext_left)     # Calculate distance
            # Draw contour for further troubleshooting
            img_1 = cv2.drawContours(crop_img.copy(), contours, -1, (0, 0, 255), 3)
            # Draw rectangle used for distance estimation
            img_out = img_1.copy()
            cv2.rectangle(img_out, (ext_left, ext_top), (ext_right, ext_bot), (0, 255, 0), 3)
            error = 0
        except Exception:
            error = 1
    else:
        return [-1, crop_img, 1]
    return [distance, img_out, error]
