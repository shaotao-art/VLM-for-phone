"""convert the grounding data to the format of box2func test
"""
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_utils import save_json

if __name__ == '__main__':
    inp_json_p = '/home/shaotao/PROJECTS/VLM_AND_PHONE/final_ablu_data/data_making/desktop_icon_elements.json'
    out_json_p = '/home/shaotao/PROJECTS/VLM_AND_PHONE/final_ablu_data/data_making/desktop_icon_elements_for_box2func_test.json'
    data = pd.read_json(inp_json_p)
    all_data = []
    for idx, row in data.iterrows():
        img_name = row['img_url']
        elements = row['element']
        os_sys = row['os']
        for ele in elements:
            all_data.append(dict(
                img_name=img_name,
                bbox=ele['bbox'],
                text=ele.get('text', 'null'),
                os=os_sys
            ))
    save_json(all_data, out_json_p)