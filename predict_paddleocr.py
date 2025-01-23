"""used for extracting text and corresponding location from images using paddleocr

one filter: filt by text"""
import os
import sys
from tqdm import tqdm
import pandas as pd
import re
from utils.file_utils import read_image, save_json

from paddleocr import PaddleOCR

def get_xyxy(input):
    """get x1, y1, x2, y2 from input"""
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp

def has_consecutive_letters(s):
    """check if the string has consecutive letters"""
    return bool(re.search(r'[A-Za-z]{2,}', s))


def filt_by_text(txts: list[str]):
    """clean and filter the text"""
    cleaned_txts = []
    for text in txts:
        text = text.strip()
        parts = text.split()
        # for icon, text type, ocr will detect icons as a single letter/number text
        if len(parts) > 0 and not has_consecutive_letters(parts[0].strip()):
            len_p0 = len(parts[0])
            text = text[len_p0:].strip()
        
        if len(parts) > 0 and not has_consecutive_letters(parts[-1].strip()):
            len_p1 = len(parts[-1])
            text = text[:-len_p1].strip()
        cleaned_txts.append(text)
    # filter out the text with length less than 2, which is usually noise
    keep_idx = [idx for idx, txt in enumerate(cleaned_txts) if len(txt) > 2]
    return cleaned_txts, keep_idx

if __name__ == '__main__':
    img_root = '/home/dmt/shao-tao-working-dir/LLM&PHONE/os-altas-desktop-merged'
    out_file_p = 'os-altas-desktop-ppocr-all.json'
    det_limit_side_len = 2560
    det_limit_type = 'max'
    
    ocr = PaddleOCR(use_angle_cls=True, 
                lang='en', 
                use_gpu=True,
                det_limit_side_len=det_limit_side_len,
                det_limit_type=det_limit_type,
                gpu_id=0)
    file_lst = os.listdir(img_root)
    file_lst = [file for file in file_lst if 'sub' not in file]
    out_lst = []
    for img_idx in tqdm(range(len(file_lst))):
    # for img_idx in tqdm(range(200)):
        try:
            img_path = os.path.join(img_root, file_lst[img_idx])
            image = read_image(img_path)
            all_result = ocr.ocr(image, cls=True)
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
        except:
            print(f'Error: {img_path}')
            continue

    # make data
    data = pd.DataFrame(out_lst)
    ground_data = []
    for idx, row in data.iterrows():
        img_path = row['img_path']
        filename = os.path.basename(img_path)
        txts = row['txts']
        boxes = row['boxes']
        cleaned_txts, keep_idx = filt_by_text(txts)
        element_lst = []
        for i, k_idx in enumerate(keep_idx):
            element_lst.append(dict(
                text=cleaned_txts[k_idx],
                bbox=boxes[k_idx],
            ))
        ground_data.append(dict(
            img_url=filename,
            element=element_lst
        ))
    save_json(ground_data, out_file_p)