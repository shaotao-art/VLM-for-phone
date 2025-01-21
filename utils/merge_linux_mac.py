import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.file_utils import read_json, save_json


def make_data(json_file_p):
    data = pd.read_json(json_file_p)
    all_data = []
    for g_name, g_data in data.groupby("img_name"):
        one_img = dict()
        one_img["img_url"] = g_name
        element_lst = []
        for _, row in g_data.iterrows():
            # print(row['text'])
            box = row['box']
            pt = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            element_lst.append({
                # "instruction": row['model_pred'],
                "instruction": None,
                "bbox": box,
                "point": pt,
                "text": row['text'],
                "type": row['type']
            })
        one_img["element"] = element_lst
        all_data.append(one_img)
    return all_data

if __name__ == '__main__':
    data_path_1 = "/home/shaotao/PROJECTS/VLM_AND_PHONE/desktop_data_making/linux_all.json"
    data_path_2 = "/home/shaotao/PROJECTS/VLM_AND_PHONE/desktop_data_making/mac_all.json"
    out_data_p = 'linux_mac_no_sampling_train_grounding.json'
    data1 = make_data(data_path_1)
    data2 = make_data(data_path_2)
    data = data1 + data2
    print('data1 len:', len(data1))
    print('data2 len:', len(data2))
    print('data len:', len(data))
    save_json(data, out_data_p)