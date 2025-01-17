import sys
import os
sys.path.append('/home/shaotao/PROJECTS/VLM_AND_PHONE')
# sys.path.append('/Users/starfish/Desktop/VLM_AND_PHONE')

from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from utils.file_utils import read_image, read_json, save_json, read_pkl



if __name__ == '__main__':
    img_root = '/home/shaotao/DATA/AMEX/screenshot'
    ann_root = '/home/shaotao/DATA/AMEX/element_anno'
    shape_pkl_p = '/home/shaotao/PROJECTS/VLM_AND_PHONE/train_val_data_making/amex/amex_img_shapes.pkl'
    df_p = '/home/shaotao/PROJECTS/VLM_AND_PHONE/train_val_data_making/amex/amex_has_func_ann.xlsx'
    ele_per_diag = 10000
    out_json_p = '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/new_format/amex_data.json'

    # img_root = ''
    # ann_root = '/Users/starfish/Downloads/element_anno'
    # shape_pkl_p = '/Users/starfish/Desktop/VLM_AND_PHONE/train_val_data_making/amex/amex_img_shapes.pkl'
    # df_p = '/Users/starfish/Desktop/VLM_AND_PHONE/train_val_data_making/amex/amex_has_func_ann.xlsx'
    # ele_per_diag = 10000
    # out_json_p = '/Users/starfish/Desktop/VLM_AND_PHONE/baseline_data/new_format/amex_data.json'

    

    df = pd.read_excel(df_p)
    df = df[df['num_has_func'] > 2]
    # get ori img shape dict
    img_shape_dict = read_pkl(shape_pkl_p)


    import random
    random.seed(42)
    all_datas = []
    
    for img_idx in tqdm(range(df.shape[0])):
        one_img = dict()
        ann_p = os.path.join(ann_root, df.iloc[img_idx]['json_file'])
        ann = read_json(ann_p)
        
        img_p = ann['image_path']
        one_img['img_url'] = img_p
        
        click_ele_lst = ann['clickable_elements']
        h, w = img_shape_dict[img_p]
        # img_p = os.path.join(img_root, img_p)
        # if img_idx % 500 == 0:
            # img = read_image(img_p)

        
        # get all clickable elements with functionality
        random.shuffle(click_ele_lst)
        final_ele_lst = []
        for ele in click_ele_lst:
            if len(final_ele_lst) == ele_per_diag:
                break
            func_ann = ele.get('functionality', '').strip()
            has_func = func_ann != ''
            if has_func:
                final_ele_lst.append(ele)
            
        element_lst = []
        for ele_idx, ele in enumerate(final_ele_lst):
            new_ele = dict()
            box = ele['bbox']
            func_ann = ele.get('functionality', '').strip()
            if func_ann.startswith('Click to '):
                func_ann = func_ann.lower().replace('click to ', '')
            new_ele['instruction'] = func_ann
            
            text_desc = ele.get('xml_desc', [''])
            if len(text_desc) > 0 and text_desc[0].strip() != '':
                new_ele['text'] = text_desc[0].replace('\n', ' ').replace('\u200b', '')
            else:
                new_ele['text'] = 'null'

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