import numpy as np
from PIL import Image
import random
from typing import List, Tuple


def get_message_infos(messages):
    img_p = None
    inst_lst = []
    pt_lst = []
    for t_idx, turn_data in enumerate(messages):
        if t_idx % 2 == 0 and t_idx == 0:
            img_p = turn_data['content'][0]['image']
            inst = turn_data['content'][1]['text']
            inst_lst.append(inst)
        elif t_idx % 2 == 0 and t_idx != 0:
            inst = turn_data['content'][0]['text']
            inst_lst.append(inst)
        elif t_idx % 2 == 1:
            pt = turn_data['content'][0]['text']
            pt_lst.append(list(eval(pt)))
    return img_p, inst_lst, pt_lst

def random_crop_metadata(img, metadata, scale_range=(0.5, 1.0)):
    original_width, original_height = img.size
    img_copy = img.copy()
    
    scale_w = random.uniform(*scale_range)
    scale_h = random.uniform(*scale_range)

    crop_width = int(original_width * scale_w)
    crop_height = int(original_height * scale_h)

    pad_x = pad_y = 0

    if crop_width > original_width or crop_height > original_height:
        pad_x = max(0, (crop_width - original_width) // 2)
        pad_y = max(0, (crop_height - original_height) // 2)

        padded_img = Image.new('RGB', (crop_width, crop_height), (255, 255, 255))
        padded_img.paste(img_copy, (pad_x, pad_y))

        img = padded_img
        img_width, img_height = crop_width, crop_height
    else:
        img_width, img_height = original_width, original_height

    crop_x_min = random.randint(0, img_width - crop_width)
    crop_y_min = random.randint(0, img_height - crop_height)
    crop_x_max = crop_x_min + crop_width
    crop_y_max = crop_y_min + crop_height

    cropped_img = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

    new_elements = []
    for element in metadata['element']:
        bbox = element['bbox']
        point = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

        bbox_abs = [int(bbox[0] * original_width) + pad_x, int(bbox[1] * original_height) + pad_y,  
                    int(bbox[2] * original_width) + pad_x, int(bbox[3] * original_height) + pad_y]
        point_abs = [int(point[0] * original_width) + pad_x, int(point[1] * original_height) + pad_y]

        if (bbox_abs[0] >= crop_x_min and bbox_abs[2] <= crop_x_max and
            bbox_abs[1] >= crop_y_min and bbox_abs[3] <= crop_y_max):
            
            new_bbox = [(bbox_abs[0] - crop_x_min) / crop_width,
                        (bbox_abs[1] - crop_y_min) / crop_height,
                        (bbox_abs[2] - crop_x_min) / crop_width,
                        (bbox_abs[3] - crop_y_min) / crop_height]
            new_point = [(point_abs[0] - crop_x_min) / crop_width,
                         (point_abs[1] - crop_y_min) / crop_height]

            new_element = element.copy()
            new_element['bbox'] = new_bbox
            new_element['point'] = new_point
            new_elements.append(new_element)

    if len(new_elements) == 0:
        return img_copy, metadata

    metadata['element'] = new_elements
    metadata['element_size'] = len(new_elements)
    metadata['img_size'] = cropped_img.size
    return cropped_img, metadata