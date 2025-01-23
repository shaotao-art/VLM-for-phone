"""sample data output from ppocr and icon detection
"""
import pandas as pd
import numpy as np
import os
import sys
from collections import Counter

from utils.file_utils import save_json

np.random.seed(42)

def check_is_chorme(text: list[str]):
    """ocr and icon detection will detect chrome tab as a single element,
    which is not useful for our task, so we need to filter out the chrome tab
    
    check if the text contains 'http'"""
    for t in text:
        if 'http' in t.lower():
            return True
    return False


def check_is_chorme_v2(elements: list[dict],
                    thres_num=5,
                    chrome_bottom=0.05,
                    chrome_left=0.05,
                    chrome_right=0.9,
                    ):
    """check if the elements are chrome tab by visual features"""
    ele_count = 0
    for ele in elements:
        box = ele['bbox']
        x1, y1, x2, y2 = box
        if ((x2 - x1) * 16) / ((y2 - y1) * 9) > 6:
            return True
        if 0 <= y1 <= chrome_bottom \
            and 0 <= y2 <= chrome_bottom \
            and chrome_left <= x1 <= chrome_right \
            and chrome_left <= x2 <= chrome_right:
            ele_count += 1
    return ele_count >= thres_num


def filt_chorme_tab(elements: list[dict],
                    is_chrome: bool,
                    chrome_bottom=0.05,
                    chrome_left=0.05,
                    chrome_right=0.9
                    ):
    if not is_chrome:
        return elements
    ele_left = []
    for ele in elements:
        box = ele['bbox']
        x1, y1, x2, y2 = box
        if is_chrome \
            and 0 <= y1 <= chrome_bottom \
            and 0 <= y2 <= chrome_bottom \
            and chrome_left <= x1 <= chrome_right \
            and chrome_left <= x2 <= chrome_right:
            continue
        ele_left.append(ele)
    return ele_left


def weighted_sampling(elements: list[dict], 
                      sample_size: int) -> list[dict]:
    """sample elements with weights by text frequency and text length"""
    text_frequency = Counter(element['text'] for element in elements)
    
    def calculate_weight(element):
        freq = text_frequency[element['text']]
        text_length = len(element['text'])
        return 1 / (freq / 2 * text_length)
    
    weights = [calculate_weight(element) for element in elements]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    indices = np.arange(len(elements))
    sampled_indices = np.random.choice(indices, 
                                       size=sample_size if sample_size < len(elements) else len(elements), 
                                       replace=False, # NOTE: no replacement
                                       p=normalized_weights)
    sampled_elements = [elements[i] for i in sampled_indices]
    return sampled_elements


if __name__ == '__main__':
    icon_data_p = '/home/shaotao/PROJECTS/VLM_AND_PHONE/final_ablu_data/os-altas-desktop-icons.json'
    ocr_data_p = '/home/shaotao/PROJECTS/VLM_AND_PHONE/final_ablu_data/os-altas-desktop-ppocr.json'
    img_root = '/home/shaotao/DATA/os-altas/desktop-merged'
    max_num_ele = 30 # max number of elements to sample
    ocr_out_p = 'desktop_ocr_elements.json'
    icon_out_p = 'desktop_icon_elements.json'

    icon_data = pd.read_json(icon_data_p)
    ocr_data = pd.read_json(ocr_data_p)
    data = pd.merge(ocr_data, icon_data, on='img_url', how='inner')
    data.rename(columns={'element_x': 'ocr_elements', 'element_y': 'icon_elements'}, inplace=True)
    print('data shape: ', data.shape)

    for idx, row in data.iterrows():
        ocr_elements = row['ocr_elements']
        icon_elements = row['icon_elements']
        text_is_chrome = check_is_chorme([ele['text'] for ele in ocr_elements])
        icon_is_chrome = check_is_chorme_v2(row['icon_elements'])
        ocr_is_chrome = check_is_chorme_v2(row['ocr_elements'])
        is_chrome = icon_is_chrome or ocr_is_chrome or text_is_chrome
        data.at[idx, 'is_chrome'] = is_chrome
        data.at[idx, 'ocr_elements'] = filt_chorme_tab(ocr_elements, is_chrome)
        data.at[idx, 'icon_elements'] = filt_chorme_tab(icon_elements, is_chrome)
        
        
    for idx, row in data.iterrows():
        ocr_elements = row['ocr_elements']
        icon_elements = row['icon_elements']
        data.at[idx, 'ocr_elements'] = weighted_sampling(ocr_elements, max_num_ele)
        data.at[idx, 'icon_elements'] = icon_elements[:max_num_ele]


    all_data = []
    num_ele = 0
    for idx, row in data.iterrows():
        img_url = row['img_url']
        element_lst = row['ocr_elements']
        all_data.append(dict(
            img_url=img_url,
            element=element_lst
        ))
        num_ele += len(element_lst)
    save_json(all_data, ocr_out_p)
    print('total number of ocr elements: ', num_ele)

    all_data = []
    num_ele = 0
    for idx, row in data.iterrows():
        img_url = row['img_url']
        element_lst = row['icon_elements']
        all_data.append(dict(
            img_url=img_url,
            element=element_lst
        ))
        num_ele += len(element_lst)
    save_json(all_data, icon_out_p)
    print('total number of icon elements: ', num_ele)