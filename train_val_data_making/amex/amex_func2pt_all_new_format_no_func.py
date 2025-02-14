"""out data format:
[
    {
        "img_url": "img_path",
        "element": [
            {
                "instruction": "click",
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "point": [0.2, 0.3],
                "text": "text"
            },
            ...
        ]
    },
    ...
]
"""
import sys
import os
PROJECT_ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from tqdm import tqdm
import pandas as pd
import random
from utils.file_utils import read_json, save_json, read_pkl

random.seed(42)


if __name__ == '__main__':
    img_root = '/home/shaotao/DATA/AMEX/screenshot'
    ann_root = '/home/shaotao/DATA/AMEX/element_anno'
    shape_pkl_p = '/home/shaotao/PROJECTS/VLM_AND_PHONE/train_val_data_making/amex/amex_img_shapes.pkl'
    df_p = '/home/shaotao/DATA/amex_no_func_20k.xlsx'
    ele_per_diag = 10000
    out_json_p = '/home/shaotao/PROJECTS/VLM_AND_PHONE/final_ablu_data/grounding_amex_only_text.json'

    df = pd.read_excel(df_p)
    # df = df.iloc[:1000]
    # get ori img shape dict
    img_shape_dict = read_pkl(shape_pkl_p)
    
    all_datas = []
    for img_idx in tqdm(range(df.shape[0])):
        one_img = dict()
        ann_p = os.path.join(ann_root, df.iloc[img_idx]['json_file'])
        ann = read_json(ann_p)
        
        img_p = ann['image_path']
        one_img['img_url'] = img_p
        
        click_ele_lst = ann['clickable_elements']
        h, w = img_shape_dict[img_p]
        
        # get all clickable elements with functionality
        random.shuffle(click_ele_lst)
        
        element_lst = []
        for ele_idx, ele in enumerate(click_ele_lst):
            new_ele = dict()
            box = ele['bbox']

            text_desc = ele.get('xml_desc', [''])
            if len(text_desc) > 0 and text_desc[0].strip() != '':
                new_ele['instruction'] = text_desc[0].replace('\n', ' ').replace('\u200b', '')
            else:
                new_ele['instruction'] = 'null'
            
            text_desc = ele.get('xml_desc', [''])
            if len(text_desc) > 0 and text_desc[0].strip() != '':
                new_ele['text'] = text_desc[0].replace('\n', ' ').replace('\u200b', '')
            else:
                new_ele['text'] = 'null'
            
            if new_ele['instruction'] == 'null' and new_ele['text'] == 'null':
                # print('no text in this element, skipping...')
                continue

            # get gt
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h
            if x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1:
                print('bbox out of range')
                print(x1, y1, x2, y2)
                break
            new_ele['bbox'] = [x1, y1, x2, y2]
            cent_x, cent_y = (x1 + x2) / 2, (y1 + y2) / 2
            pt = [cent_x, cent_y]
            new_ele['point'] = pt
            element_lst.append(new_ele)
        if len(element_lst) == 0:
            print('no functionality in this image, skipping...')
            continue
        one_img['element'] = element_lst
        all_datas.append(one_img)

    print('total data num: ', len(all_datas))
    save_json(all_datas, out_json_p)