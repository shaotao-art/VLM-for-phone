
import os
import sys
PROJECT_ROOT = '/home/shaotao/PROJECTS/VLM_AND_PHONE/'
sys.path.append(PROJECT_ROOT)


import json
import pandas as pd
from tabulate import tabulate
import argparse

from utils.file_utils import read_json, read_image, save_image, save_json
from utils.helper_utils import print_args
from eval.aitw.action_matching import action_matching



def parse_model_output(output_string):
    try:
        out = json.loads(output_string)
        # action = out['action']
        # # only click and type has value
        # value = out.get('value', '')

        action = out['action_type']
        # only click and type has value
        value = out.get('action_value', '')
    except Exception as e:
        print(f"Error in parsing model output: {output_string}")
        print(e)
        action, value = 'error', ''
    return action, value
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate action matching results.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the JSON data file.')
    args = parser.parse_args()
    print_args(args)
    
    data_p = args.data_path
    data = read_json(data_p)

    res = []
    for sample in data:
        img_p = sample['image_lst'][0]
        assert sample['conversation'][1]['from'] == 'gpt'
        assert sample['conversation'][2]['from'] == 'prediction'
        gt = sample['conversation'][1]['value']
        pred = sample['conversation'][2]['value']

        try:
            gt_action, gt_args = parse_model_output(gt)
            pred_action, pred_args = parse_model_output(pred)
            assert pred_action.upper() in ['ERROR', 'SCROLL_UP', 'SCROLL_DOWN', 'SCROLL_LEFT', 'SCROLL_RIGHT', 'CLICK', 'TYPE', 'TASK_COMPLETE', 'TASK_IMPOSSIBLE', 'HOME', 'ENTER', 'BACK']
        except Exception as e:
            print(e)
            print(f"Error in parsing action: {gt}, {pred}")
            continue
        
        try:
            match_res = action_matching(pred_action.upper(), 
                                        pred_args, 
                                        gt_action.upper(), 
                                        gt_args)
        except:
            print(f"Error in matching action: {pred_action}, {gt_action}, {pred_args}, {gt_args}")
            continue
        match_res['img_meta'] = img_p
        match_res['pred_action'] = pred_action
        match_res['pred_args'] = pred_args
        match_res['gt_action'] = gt_action
        match_res['gt_args'] = gt_args
        res.append(match_res)


    df = pd.DataFrame(res)
    df['split'] = df['img_meta'].apply(lambda x: x.split('/')[-2])
    def get_acc(df):
        if len(df) == 0:
            return '[0/0]-1'
        num_right = (df['is_correct'] == 'yes').sum()
        num_total = len(df)
        acc = num_right / num_total
        return f'[{num_right}/{num_total}]\n{round(acc * 100, 1)}'

    split_lst = ['general', 'install', 'googleapps', 'single', 'webshopping']
    action_lst = ['click', 'scroll_up', 'scroll_down', 'scroll_left', 'scroll_right', 'type', 'home', 'enter', 'back', 'task_complete', 'task_impossible']

    cm_data = [
        ["", 'general', 'install', 'googleapps', 'single', 'webshopping', 'overall'],
        ["click", '', '', '', '', '', ''],
        ["scroll_up", '', '', '', '', '', ''],
        ["scroll_down", '', '', '', '', '', ''],
        ["scroll_left", '', '', '', '', '', ''],
        ["scroll_right", '', '', '', '', '', ''],
        ["type", '', '', '', '', '', ''],
        ["home", '', '', '', '', '', ''],
        ["enter", '', '', '', '', '', ''],
        ["back", '', '', '', '', '', ''],
        ["task_complete", '', '', '', '', '', ''],
        ["task_impossible", '', '', '', '', '', ''],
        ["overall", '', '', '', '', '', ''],
    ]

    for s in split_lst:
        filt = df['split'] == s
        cm_data[-1][split_lst.index(s) + 1] = get_acc(df[filt])
        for a in action_lst:
            total_filt = (df['split'] == s) & (df['gt_action'] == a)
            # action do not match
            filt = (df['pred_action'] != a) & (df['gt_action'] == a)
            filt = filt & (df['split'] == s)
            cm_data[action_lst.index(a) + 1][split_lst.index(s) + 1] = f'[{filt.sum()}/{total_filt.sum()}]{-1 if total_filt.sum() == 0 else round(filt.sum() / total_filt.sum() * 100, 2)}'
            
            # action match but args not match
            filt = (df['pred_action'] == a) & (df['gt_action'] == a)
            filt = filt & (df['split'] == s)
            cm_data[action_lst.index(a) + 1][split_lst.index(s) + 1] += f'\n{get_acc(df[filt])}'
            

    for a in action_lst:
        filt = df['gt_action'] == a
        cm_data[action_lst.index(a) + 1][-1] = get_acc(df[filt])

    cm_data[-1][-1] = get_acc(df)
    print('value1: action not match ratio \nvalue2: action match but args not match ratio')
    print(tabulate(cm_data, headers='firstrow', tablefmt='grid'))

    out_file_p = data_p.replace('.json', '_eval.json')
    res = {
        'general': cm_data[-1][1],
        'install': cm_data[-1][2],
        'googleapps': cm_data[-1][3],
        'single': cm_data[-1][4],
        'webshopping': cm_data[-1][5],
        'overall': cm_data[-1][-1],
    }
    # save into json file
    save_json(res, out_file_p)

    df = pd.DataFrame(cm_data[1:], columns=cm_data[0])
    df.to_csv(out_file_p.replace('.json', '.csv'), index=False)

