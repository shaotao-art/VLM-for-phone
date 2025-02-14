import json
import numpy as np
import os
from copy import deepcopy
from tqdm import tqdm
import ast
from typing import Tuple, Dict, List
import sys
import argparse
import json
import pandas as pd
sys.path.append('/home/shaotao/PROJECTS/VLM_AND_PHONE')
# sys.path.append('/Users/starfish/Desktop/VLM_AND_PHONE')


from utils.file_utils import read_json, save_json, read_pkl
from utils.helper_utils import print_args
from prompts import all_prompts

def format_his_info(action_his:List[str]):
    act_his_str = ''
    for his in action_his:
        # print(type(his))
        # print(his)
        his_dict = ast.literal_eval(his)
        # print(type(his_dict))
        if 'action' in his_dict:
            act_his_str += f"action: {his_dict['action']} "
        if 'text' in his_dict:
            # for type and select
            act_his_str += f"action_type: {his_dict['action_type']}, point: {his_dict['point']}, text: {his_dict['text']}\n"
        else:
            # for click
            act_his_str += f"action_type: {his_dict['action_type']}, point: {his_dict['point']}\n"
    return act_his_str.strip()

def parse_action_type(ann):
    img_height, img_width = img_shapes[filename]
    text = ann["operation"]["value"]
    action_type = ann["operation"]["original_op"]
    point_x = ann["bbox"]["x"] + (ann["bbox"]["width"] / 2)
    point_y = ann["bbox"]["y"] + (ann["bbox"]["height"] / 2)
    point_x = point_x / img_width
    point_y = point_y / img_height
    pt = (int(point_x * 1000), int(point_y * 1000))
    if action_type in ['CLICK', 'HOVER', 'ENTER']:
        answer = dict(
            action_type='click',
            point=pt
        )
    elif action_type == 'SELECT':
        answer = dict(
            action_type='select',
            point=pt,
            text=text
        )
    elif action_type == 'TYPE':
        answer = dict(
            action_type='type',
            point=pt,
            text=text
        )
        
    else:
        raise ValueError(f'Unknown action type: {action_type}')
    
    return answer

def make_test_gt(ann):
    img_height, img_width = img_shapes[filename]
    text = ann["operation"]["value"]
    action_type = ann["operation"]["original_op"]
    box = [ann["bbox"]["x"], ann["bbox"]["y"], ann["bbox"]["width"], ann["bbox"]["height"]]
    box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
    box = box[0] / img_width, box[1] / img_height, box[2] / img_width, box[3] / img_height
    box = list(map(lambda x: int(x * 1000), box))
    gt = dict(
        action_type=mind2webaction2action[action_type],
        box=box,
        text=text,
        ann_id=annot_id
    )
    return gt




def parse_args():
    parser = argparse.ArgumentParser(description="Process mind2web data")
    parser.add_argument('--img_root', type=str, help='Root directory of images')
    parser.add_argument('--inp_json_p', type=str, help='Input JSON file path')
    parser.add_argument('--out_json_p', type=str, help='Output JSON file path')
    parser.add_argument('--img_shape_pkl_p', type=str, help='Image shapes pickle file path')
    parser.add_argument('--hist_len', type=int, help='Length of the action history')
    parser.add_argument('--cot_ann_p', type=str, default=None, help='multi level annotation json path')
    parser.add_argument('--answer_cot_level', type=str, default=None, help='cot annotation level')
    parser.add_argument('--hist_use_action', action='store_true', help='use action in history')
    return parser.parse_args()


mind2webaction2action = dict(
    CLICK='click',
    HOVER='click',
    ENTER='click',
    SELECT='select',
    TYPE='type'
)    


if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    img_root = args.img_root
    inp_json_p = args.inp_json_p
    out_json_p = args.out_json_p
    img_shape_pkl_p = args.img_shape_pkl_p
    hist_len = args.hist_len
    cot_ann_p = args.cot_ann_p
    answer_cot_level = args.answer_cot_level
    hist_use_action = args.hist_use_action
    
    # img_root = '/home/shaotao/DATA/mind2web/ming2web_images'
    # inp_json_p = '/home/shaotao/DATA/mind2web/mind2web_data_train.json'
    # img_shape_pkl_p = '/home/shaotao/PROJECTS/VLM_AND_PHONE/train_val_data_making/agent_task/mind2web_img_shapes.pkl'
    # cot_ann_p = '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/mind2web-l3.xlsx'
    # out_json_p = 'mind2web-naive.json'
    # hist_len = 4
    # # cot_level = 'action'
    # cot_ann_p = None
    # cot_level = None

    if answer_cot_level is not None:
        assert answer_cot_level in ['action', 'thoughts', 'null']
    
    all_ann = read_json(inp_json_p)
    img_shapes = read_pkl(img_shape_pkl_p)
    PROMPT = all_prompts['agent_action_caption_mind2web']
    if cot_ann_p is not None:
        cot_ann = pd.read_excel(cot_ann_p)
        cot_ann.set_index('img_name', inplace=True)
    

    all_data = []
    for p_idx, sample in enumerate(tqdm(all_ann)):
        # one trajectory
        instruction = sample['confirmed_task']
        action_lst = sample['actions']
        annot_id = sample['annotation_id']

        image_lst = []
        action_his = []
        for step_idx in range(len(action_lst)):
            # one step
            conversation = []
            ann = action_lst[step_idx]
            filename = annot_id + '-' + ann["action_uid"] + '.jpg'
            img_filename = os.path.join(img_root, filename)
            if os.path.isfile(img_filename) == False:
                # print('one file do not exist')
                continue
        
            cot = None
            if cot_ann_p is not None:
                try:
                    multi_level_ann_line = cot_ann.loc[filename]
                except:
                    # print('one img do not have multi-level ann, skipping...')
                    continue
                # previous_actions = multi_level_ann_line['previous_actions']
                observation = multi_level_ann_line['observation']
                thought = multi_level_ann_line['thought']
                action = multi_level_ann_line['action']
                cot = dict(
                    observation=observation,
                    thought=thought,
                    action=action
                )
            # history format: action(optional), action_type, action_params   
            act_his_str = format_his_info(action_his[-hist_len:])
            if len(action_his) == 0:
                act_his_str = 'null'
            
            
            
            plain_answer = parse_action_type(ann)
            # print('plain answer: ', plain_answer)
            if cot is not None:
                if answer_cot_level == 'action':
                    answer = {'action': cot['action']}
                    answer.update(plain_answer)
                elif answer_cot_level == 'thoughts':
                    answer = deepcopy(cot)
                    answer.update(plain_answer)
                elif answer_cot_level == 'null':
                    answer = plain_answer                    
            else:
                answer = plain_answer
            # print('answer: ', answer)
            
            prompt = PROMPT.format(instruction=instruction, 
                                   action_history=act_his_str,
                                   action_type=answer['action_type'],
                                   point=answer['point'],
                                   text=answer.get('text', 'null'))
            if '<image>' not in prompt:
                prompt = '<image>' + prompt
            
            
            # history format: action(optional), action_type, action_params   
            if hist_use_action:
                hist_line = {'action': cot['action']}
                hist_line.update(plain_answer)
            else:
                hist_line = plain_answer
            # print('hist line: ', hist_line)
            action_his.append(str(hist_line))
            conversation.append({'from': 'human', 'value': prompt})
            if 'test' not in inp_json_p:
                conversation.append({'from': 'gpt', 'value': cot['action']})

            line = {'conversation': conversation, 'image_lst': [img_filename]}
            all_data.append(line)
            if len(all_data) == 2:
                print(f'>>> sample: {line}')
        # break
    print(f'>>> total data num: {len(all_data)}')
    save_json(all_data, out_json_p)