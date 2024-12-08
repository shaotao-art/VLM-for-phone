"""get amex's number of description and functionality annotation for """
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath('../'))
sys.path.append(PROJECT_ROOT)

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils.file_utils import read_json


def process_file(idx):
    filename = file_lst[idx]
    ann = read_json(os.path.join(ann_root, filename))
    click_ele_lst = ann['clickable_elements']

    num_desc_ann = 0
    num_func_ann = 0
    for ele in click_ele_lst:
        desc = ele.get('xml_desc', [])
        if len(desc) > 0:
            num_desc_ann += 1
            
        func_ann = ele.get('functionality', '')
        if len(func_ann) > 0:
            num_func_ann += 1

    return dict(filename=filename, num_desc_ann=num_desc_ann, num_func_ann=num_func_ann)

if __name__ == '__main__':
    ann_root = '/home/shaotao/DATA/AMEX/element_anno'
    
    file_lst = sorted(os.listdir(ann_root))
    all_data = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(process_file, range(len(file_lst))), total=len(file_lst)))

    all_data.extend(results)
    # save all data into a csv file
    df = pd.DataFrame(all_data)
    df.to_excel('amex_info.xlsx', index=False)