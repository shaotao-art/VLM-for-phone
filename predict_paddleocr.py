"""used for extracting text and corresponding location from images using paddleocr"""
import sys
import os
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

sys.path.append('/home/shaotao/PROJECTS/VLM_AND_PHONE/')
from utils.draw_utils import draw_box, draw_text
from utils.file_utils import read_image, save_image, save_json


def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp

def has_more_than_two_letters(s, threshold=2):
    letter_count = 0
    for char in s:
        if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
            letter_count += 1
            if letter_count > threshold:
                return True
    return letter_count > threshold


if __name__ == '__main__':
    # img_root = '/home/shaotao/DATA/os-altas/os-altas-macos/'
    # out_file_p = '/home/shaotao/DATA/os-altas/os-altas-macos-ppocr.json'
    # vis_root = './vis_mac'
    # det_limit_side_len = 960
    # vis = False
    
    
    img_root = '/home/shaotao/DATA/os-altas/os-altas-linux/'
    out_file_p = '/home/shaotao/DATA/os-altas/os-altas-linux-ppocr.json'
    vis_root = './vis_linux'
    det_limit_side_len = 1280
    vis = False
    
    
    os.makedirs(vis_root, exist_ok=True)
    ocr = PaddleOCR(use_angle_cls=True, 
                lang='en', 
                use_gpu=True,
                gpu_mem=24000,
                det_limit_side_len=det_limit_side_len,
                gpu_id=3)
    file_lst = os.listdir(img_root)
    file_lst = [file for file in file_lst if 'sub' not in file]
    out_lst = []
    for img_idx in tqdm(range(len(file_lst))):
        # for img_idx in range(10):
        try:
            img_path = os.path.join(img_root, file_lst[img_idx])
            image = read_image(img_path)
            all_result = ocr.ocr(image, cls=True)
            # draw result
            result = all_result[0]
            h, w = image.shape[:2]
            # normalize the box
            boxes_01 = [get_xyxy(line[0]) for line in result]
            boxes_01 = [[box[0]/w, box[1]/h, box[2]/w, box[3]/h] for box in boxes_01]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            out_lst.append(dict(
                img_path=img_path,
                txts=txts,
                boxes=boxes_01,
                scores=scores
            ))
            if vis:
                for idx, box_ in enumerate(boxes_01):
                    image = draw_box(image, box_)
                    image = draw_text(image, txts[idx], box_[2:])
                save_image(image, os.path.join(vis_root, file_lst[img_idx]))
        except:
            print(f'Error: {img_path}')
            continue
    save_json(out_lst, out_file_p)
    





