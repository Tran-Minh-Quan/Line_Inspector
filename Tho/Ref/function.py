import numpy as np


def find_crop_coordinate(bounding_box_file, object_width, focal_length, image_width, image_height):
    bounding_data = np.loadtxt(bounding_box_file)
    x_center = bounding_data[1]*image_width
    y_center = bounding_data[2]*image_height
    width_pixel = bounding_data[3]*image_width
    height_pixel = bounding_data[4] * image_height
    #distance = int((10*(focal_length*object_width)/width_pixel)+0.5)
    start_point = [int(x_center-width_pixel/2), int(y_center-height_pixel/2)]
    end_point = [int(x_center+width_pixel/2), int(y_center+height_pixel/2)]
    return [start_point, end_point]


def calculate_distance(object_width, focal_length, width_pixel):
    distance = int((10 * (focal_length * object_width) / width_pixel) + 0.5)
    return distance
