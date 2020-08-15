import imutils
import cv2
import numpy as np


def calculate_distance(object_width, focal_length, width_pixel):
    distance = int((10 * (focal_length * object_width) / width_pixel) + 0.5)
    return distance


def distance_estimate(img, start_point, end_point, extended_ratio, canny_var_1, canny_var_2, object_width):
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
            distance = calculate_distance(object_width, 764, ext_right - ext_left)     # Calculate distance
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


def distance_estimate_developing(img, start_point, end_point, extended_ratio, canny_var_1, canny_var_2):
    kernel = np.ones((5, 5), np.uint8)  # Init kernel for erosion operation
    width_box = end_point[1] - start_point[1]   # Calculate width of bounding box
    height_box = end_point[0] - start_point[0]  # Calculate height of bounding box
    # Calculate x axis extended
    x_axis_extended = [max(0, int(start_point[1] - extended_ratio * width_box)),
                       min(img.shape[0], int(end_point[1] + extended_ratio * width_box))]
    # Calculate y axis extended
    y_axis_extended = [max(0, int(start_point[0] - extended_ratio * height_box)),
                       min(img.shape[1], int(end_point[0] + extended_ratio * height_box))]
    # Crop image
    crop_img = img[x_axis_extended[0]: x_axis_extended[1], y_axis_extended[0]:y_axis_extended[1]]
    img_1 = crop_img.copy()
    erode = cv2.erode(crop_img, kernel, iterations=1)   # Perform erosion operation on input image
    edges = cv2.Canny(erode, canny_var_1+10, canny_var_2+10)  # Perform Canny edge detection algorithm on eroded image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   # Find contours of the image
    """contours = imutils.grab_contours(contours)  # Grabs the appropriate tuple value
    c = sorted(contours, key=cv2.contourArea)   # Sort contours based on contour area
    ext_left, ext_right, ext_top, ext_bot = 1e9, -1e9, 1e9, -1e9    # Initialize external boundary on contours
    # Find external boundary of contours
    for cc in c:
        ext_left = min(cc[:, :, 0].min(), ext_left)
        ext_right = max(cc[:, :, 0].max(), ext_right)
        ext_top = min(cc[:, :, 1].min(), ext_top)
        ext_bot = max(cc[:, :, 1].max(), ext_bot)
    distance = calculate_distance(7, 764, ext_right - ext_left)     # Calculate distance
    # Draw contour for further troubleshooting
    img_1 = cv2.drawContours(crop_img.copy(), contours, -1, (0, 0, 255), 3)
    # Draw rectangle used for distance estimation
    cv2.rectangle(img_1, (ext_left, ext_top), (ext_right, ext_bot), (0, 255, 0), 3)"""
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 2:
            cv2.drawContours(img_1, contour, -1, (0, 0, 255), 3)

            c = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.drawContours(img_1, contours, -1, (0, 0, 255), 3)
    distance = 1
    return [distance, img_1]
