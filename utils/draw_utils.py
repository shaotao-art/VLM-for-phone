import os
import sys
PROJECT_ROOT=os.path.dirname(os.path.abspath('.'))
sys.path.append(PROJECT_ROOT)
import cv2
import ast
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from typing import Tuple, List, Callable


from utils.file_utils import save_image




class default_draw_config:
    line_color = (255, 165, 0)
    line_thickness = 10

    box_color = (255, 165, 0)
    box_thickness = 10
    
    arrow_color = (255, 165, 0)
    arrow_thickness = 10
    arrow_tipLength = 0.1
    
    font_color = (255, 0, 0)
    font_scale = 1.0
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    dot_color = (255, 0, 0)
    dot_radius = 40
    dot_thickness = cv2.FILLED
    
    dash_length = 10 
    space_length = 20
    
    transparent_alpha = 0.5
    
def transparent_decorator(alpha: float = default_draw_config.transparent_alpha):    
    def decorator(func: Callable):
        def wrapper(img: np.ndarray, *args, **kwargs) -> np.ndarray:
            overlay = deepcopy(img)
            overlay = func(overlay, *args, **kwargs)
            output = deepcopy(img)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
            return output
        return wrapper
    return decorator


def float_cord_to_int_xy(pt, img: np.ndarray):
    height, width = img.shape[:2]
    if len(pt) == 2:
        x, y = pt
        x = int(x * width)
        y = int(y * height)
        return x, y
    elif len(pt) == 4:
        x1, y1, x2, y2 = pt
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        return x1, y1, x2, y2
    else:
        raise ValueError('Invalid input') 


def show_image(img: np.ndarray):
    plt.imshow(img)
    plt.axis('off')
    plt.show()


@transparent_decorator()
def draw_dot(img: np.ndarray, 
             pt: Tuple[float, float],
             color=default_draw_config.dot_color, 
             radius=default_draw_config.dot_radius, 
             thickness=default_draw_config.dot_thickness):
    img_ = deepcopy(img)
    pt = float_cord_to_int_xy(pt, img)
    cv2.circle(img_, pt, radius, color, thickness)
    return img_

@transparent_decorator()
def draw_arrow(img: np.ndarray, 
               pt1: Tuple[float, float],
               pt2: Tuple[float, float],
               color=default_draw_config.arrow_color, 
               thickness=default_draw_config.arrow_thickness, 
               tipLength=default_draw_config.arrow_tipLength):
    img_ = deepcopy(img)
    pt = float_cord_to_int_xy(pt, img)
    cv2.arrowedLine(img_, pt1, pt2, color, thickness, tipLength=tipLength)
    return img_

@transparent_decorator()
def draw_box(img: np.ndarray, 
             box: Tuple[float, float, float, float], 
             color=default_draw_config.box_color, 
             thickness=default_draw_config.box_thickness):
    img_ = img.copy()
    box = float_cord_to_int_xy(box, img)
    cv2.rectangle(img_, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return img_

@transparent_decorator()
def draw_text(img: np.ndarray, 
              text: str, 
              pt: Tuple[int, int], 
              color=default_draw_config.font_color, 
              scale=default_draw_config.font_scale, 
              thickness=default_draw_config.font_thickness, 
              font=default_draw_config.font):
    img_ = img.copy()
    cv2.putText(img_, text, pt, font, scale, color, thickness, cv2.LINE_AA)
    return img_