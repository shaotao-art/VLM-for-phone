import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath('..'))
sys.path.append(PROJECT_ROOT)

from utils.file_utils import read_json

import logging
import ast
import pandas as pd
from tabulate import tabulate
import argparse

logging.basicConfig(level=logging.INFO)


def to_float_box(bbox, img_h, img_w):
    if type(bbox) == str:
        bbox = ast.literal_eval(bbox)
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    bbox = [bbox[0] / img_w, 
            bbox[1] / img_h, 
            bbox[2] / img_w, 
            bbox[3] / img_h]
    return bbox

def judge(row):
    try:
        pred_action = row['action']
        pred_args = row['args']
        gt = row['float_bbox']
        if pred_action == 'click':
            pt = ast.literal_eval(pred_args)
            assert len(pt) == 2
            x, y = pt[0] / 1000, pt[1] / 1000
            return (x >= gt[0]) and (x <= gt[2]) and (y >= gt[1]) and (y <= gt[3])
        else:
            return False
    except:
        logging.error(f"Error in judge: {row['action']}, {row['args']}")
    
def get_acc(df):
    if len(df) == 0:
        return '[0/0]-1'
    num_right = (df['is_correct'] == True).sum()
    num_total = len(df)
    acc = num_right / num_total
    return f'[{num_right}/{num_total}]\n{round(acc * 100, 1)}'


def get_pt_yinhao(x):
    """expect format: 'click:(x, y)'"""
    return x.split(':')[-1].strip()

def get_pt_kuohao(x):
    """expect format: 'click((x, y))'"""
    # find the first '('
    start = x.find('(')
    # find the last ')'
    end = x.rfind(')')
    return x[start + 1:end].strip()

def get_pt_indentity(inp):
    """expect format: '(x, y)'"""
    return inp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate screen spot predictions.')
    parser.add_argument('--res_file_p', type=str, help='Path to the result JSON file')
    parser.add_argument('--method', type=str, help='Model output parse method')
    args = parser.parse_args()

    res_file_p = args.res_file_p
    m = args.method
    print('input file:', res_file_p)
    
    data = read_json(res_file_p)
    
    parser_str2method = {
        'yinhao': get_pt_yinhao,
        'kuohao': get_pt_kuohao,
        'identity': get_pt_indentity
    }
    method = parser_str2method[m]

    df = pd.DataFrame(data)
    df['args'] = df['model_pred'].apply(lambda x: method(x))
    df['float_bbox'] = df.apply(lambda x: to_float_box(x['bbox'], 
                                                    x['ori_img_shape'][0], 
                                                    x['ori_img_shape'][1]), axis=1)
    df['is_correct'] = df.apply(judge, axis=1)
    df['device'] = df['ori_img_shape'].apply(lambda x: 'phone' if x[0] > x[1] else 'pad')


    cm_data = [
        ["", "phone", "pad", "overall"],
        ["text", '', '', ''],
        ["icon", '', '', ''],
        ['overall', '', '', '']
    ]

    type_lst = ['text', 'icon']
    device_lst = ['phone', 'pad']
    for s in type_lst:
        for d in device_lst:
            acc = get_acc(df[(df['data_type'] == s) & (df['device'] == d)])
            cm_data[type_lst.index(s) + 1][device_lst.index(d) + 1] = acc
        acc = get_acc(df[df['data_type'] == s])
        cm_data[type_lst.index(s) + 1][3] = acc
    for d in device_lst:
        acc = get_acc(df[df['device'] == d])
        cm_data[3][device_lst.index(d) + 1] = acc
    acc = get_acc(df)
    cm_data[3][3] = acc
    print(tabulate(cm_data, headers='firstrow', tablefmt='grid'))