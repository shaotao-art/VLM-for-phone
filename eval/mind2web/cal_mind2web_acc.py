import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath('..'))
sys.path.append(PROJECT_ROOT)

import logging
import numpy as np
import json
import pandas as pd
import tabulate
import argparse

from utils.file_utils import read_json
from utils.helper_utils import print_args



def calculate_f1(pred, label):
    """calculate action f1 following mind2web
    """
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def parse_model_output(output_string):
    try:
        out = json.loads(output_string)
        # action and pt is a must in mind2web
        action = out['action_type']
        pt = out['point']
        text = out.get('text', '')
    except Exception as e:
        print(f"Error in parsing model output: {output_string}")
        # print(e)
        action, pt, text = 'error', [-1, -1], ''
    return action, pt, text


def get_metrices(row):
    try:
        pred_action, pred_pt, pred_text = row['pred_action'], row['pred_pt'], row['pred_text']
        gt_action, gt_box, gt_text = row['gt_action'], row['gt_box'], row['gt_text']
        if pred_action == gt_action:
            row['Op_match'] = True
        if (gt_box[0] <= pred_pt[0] <= gt_box[2]) and (gt_box[1] <= pred_pt[1] <= gt_box[3]):
            row['Ele_match'] = True
                
        pred_str = pred_action
        if pred_action == 'select' or pred_action == 'type':
            pred_str += ' '
            pred_str += pred_text.lower()
        ref_str = gt_action
        if gt_action == 'select' or gt_action == 'type':
            ref_str += ' '
            ref_str += gt_text.lower()
        f1 = calculate_f1(pred_str, ref_str)
        row['Op_f1'] = f1
        if f1 == 1.0 and row['Ele_match'] == True:
            row['step_success'] = 1.0
    
    except Exception as e:
        logging.error(f"Error in matching action: {pred_action}, {gt_action}")
        logging.error(e)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate action matching results.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the JSON data file.')
    args = parser.parse_args()
    print_args(args)

    data_p = args.data_path
    data = read_json(data_p)

    # make df
    res = []
    for sample in data:
        img_p = sample['image_lst'][0]
        assert sample['conversation'][1]['from'] == 'gpt'
        assert sample['conversation'][2]['from'] == 'gt'
        assert sample['conversation'][3]['from'] == 'prediction'
        gt = sample['conversation'][2]['value']
        pred = sample['conversation'][3]['value']

        try:
            gt_action, gt_box, gt_text = gt['action_type'], gt['box'], gt['text']
            gt_box = [int(x) for x in gt_box]
            pred_action, pred_pt, pred_text = parse_model_output(pred)
            assert pred_action.upper() in ['ERROR', 'SCROLL', 'CLICK', 'TYPE', 'TASK_COMPLETE', 'SELECT']
        except:
            logging.error(f"Error in parsing action: {gt}, {pred}")
            continue
        
        match_res = {}
        match_res['img_meta'] = img_p
        match_res['ann_id'] = gt['ann_id']
        match_res['pred_action'] = pred_action.strip()
        match_res['pred_pt'] = pred_pt
        match_res['pred_text'] = pred_text.strip()
        match_res['gt_action'] = gt_action.strip()
        match_res['gt_box'] = gt_box
        match_res['gt_text'] = gt_text.strip()
        res.append(match_res)


    # cal acc
    df = pd.DataFrame(res)
    df['Op_match'] = False
    df['Ele_match'] = False 
    df['Op_f1'] = 0
    df['step_success'] = 0
    
    df = df.apply(get_metrices, axis=1)
    step_sr = df['step_success'].mean()
    op_match_acc = df['Op_match'].mean()
    ele_acc = df['Ele_match'].mean()
    op_f_lst = []
    for g_name, g_df in df.groupby('ann_id'):
        g_f1 = g_df['Op_f1'].mean()
        op_f_lst.append(g_f1)
    op_f1 = np.mean(op_f_lst)

    ele_acc = round(ele_acc * 100, 1)
    op_f1 = round(op_f1 * 100, 1)
    step_sr = round(step_sr * 100, 1)
    op_match_acc = round(op_match_acc * 100, 1)
    table = [['ele_acc', ele_acc],
            ['op_f1', op_f1],
            ['step_success', step_sr], 
            ['op_match_acc', op_match_acc]]
    print(tabulate.tabulate(table, tablefmt='grid'))

    # save result to json
    res = {
        'ele_acc': ele_acc,
        'op_f1': op_f1,
        'step_success': step_sr,
        'op_match_acc': op_match_acc
    }
    res_p = data_p.replace('.json', '_res.json')
    with open(res_p, 'w') as f:
        json.dump(res, f)





