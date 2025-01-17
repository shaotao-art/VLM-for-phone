import pandas as pd
import os
import sys
from tqdm import tqdm

sys.path.append('/home/shaotao/PROJECTS/VLM_AND_PHONE')
from utils.file_utils import save_json


def get_if_click(line):
    try:
        return line[0]['name'] == 'click'
    except:
        print('error:', line)
        return False
    
if __name__ == '__main__':
    json_p = '/home/shaotao/DATA/GUIAct/web-single_train_data.json'
    img_root = '/home/shaotao/DATA/GUIAct/guiact-single'
    out_json_p = 'gui-act-single-all-new-format.json'
    df = pd.read_json(json_p)
    # get only clickable elements
    df['clickable'] = df['actions_label'].apply(get_if_click)
    df = df[df['clickable'] == True]
    all_groups = df.groupby('image_id')
    
    total_ele_num = 0
    all_datas = []    
    for g_idx, (img_id, g_df) in enumerate(tqdm(all_groups)):
        if len(g_df) == 0:
            print('empty group, skip')
            continue
        one_img = {}
        total_ele_num += len(g_df)
        img_filename = f'{img_id}.jpg'
        one_img['img_url'] = img_filename
        img_path = os.path.join(img_root, img_filename)
        
        # shuffle all elements
        g_df = g_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        element_lst = []
        for i_idx, row in g_df.iterrows():
            new_ele = {}
            # get input prompt
            question = row['question']
            # get answer
            ann = row['actions_label'][0]
            box = ann['element']['related'].split('<box>')[1].split('</box>')[0]
            box = [float(_) for _ in box.split(',')]
            pt = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            new_ele['instruction'] = question
            new_ele['bbox'] = box
            new_ele['point'] = pt
            element_lst.append(new_ele)
        assert len(element_lst) > 0
        one_img['element'] = element_lst
        all_datas.append(one_img)

    print('total data num: ', len(all_datas))
    print('sample data: ', all_datas[0])
    print('total ele num: ', total_ele_num)
    save_json(all_datas, out_json_p)