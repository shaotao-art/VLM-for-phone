import cv2
import numpy as np

from utils.helper_utils import smart_resize

def resize_image_short_side(image_array: np.ndarray, 
                            x: int):
    height, width = image_array.shape[:2]
    
    if width <= height:
        new_width = x
        new_height = int(height * (x / width))
    else:
        new_width = int(width * (x / height))
        new_height = x
    
    img_resized = cv2.resize(image_array, 
                             (new_width, new_height), 
                             interpolation=cv2.INTER_AREA)
    return img_resized

def resize_image_max_side(image_array: np.ndarray, 
                          x: int):
    height, width = image_array.shape[:2]
    
    if width <= height:
        new_height = x
        new_width = int(width * (x / height))
    else:
        new_width = x
        new_height = int(height * (x / width))
    
    img_resized = cv2.resize(image_array, 
                             (new_width, new_height), 
                             interpolation=cv2.INTER_AREA)
    return img_resized


def smart_resize_img(image_array: np.ndarray, 
                     max_pixels: int):
    height, width = image_array.shape[:2]
    new_height, new_width = smart_resize(height, 
                                         width, 
                                         max_pixels)
    img_resized = cv2.resize(image_array, 
                             (new_width, new_height), 
                             interpolation=cv2.INTER_AREA)
    return img_resized
