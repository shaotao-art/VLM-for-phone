import json
import numpy as np
import os
import ast
import sys
sys.path.append(os.path.dirname(os.path.abspath('.')))
from utils.file_utils import read_image, read_json, save_image, save_json
from utils.helper_utils import float2_0_1000, print_args
from prompts import all_prompts
import argparse
import json
from typing import Tuple



def determine_swipe_direction(str_pt: Tuple[float, float], 
                              end_pt: Tuple[float, float]) -> str:
    delta_x = end_pt[0] - str_pt[0]
    delta_y = end_pt[1] - str_pt[1]

    if abs(delta_x) > abs(delta_y):
        if delta_x > 0:
            return 'right'
        else:
            return 'left'
    else:
        if delta_y > 0:
            return 'down'
        else:
            return 'up'

def parse_action_type(ann):
    action_type_id = int(ann['action_type_id'])
    action_type_text = ann["action_type_text"]
    touch_pt = ann['touch']
    lift_pt = ann['lift']
    text = ann['type_text'].replace('"', "'")
    
    # 4: click
    # 3: type
    # 10: task complete
    # 11: task impossible
    # 6: home
    # 5: back
    # 7: enter
    
    if action_type_id == 4:
        if action_type_text == 'click':
            cent_x = (touch_pt[0] + lift_pt[0]) / 2
            cent_y = (touch_pt[1] + lift_pt[1]) / 2
            cnet_x = float2_0_1000(cent_x)
            cnet_y = float2_0_1000(cent_y)
            content = f'{{\"action\":\"click\",\"value\":[{cnet_x},{cnet_y}]}}'
        # NOTE: action in ori data maybe wrong
        elif action_type_text in ['scroll up', 'scroll down', 'scroll left', 'scroll right']:
            direction = determine_swipe_direction(touch_pt, lift_pt)
            content = f'{{\"action\":\"scroll_{direction}\"}}'
        else:
            print('error click')
            print(ann)
            content = 'error'
        
        
    elif action_type_id == 3:
        content = f'{{\"action\":\"type\",\"value\":\"{text}\"}}'
        
    elif action_type_id == 10:
        content = '{\"action\":\"task_complete\"}'
        
    elif action_type_id == 6:
        content = '{\"action\":\"home\"}'

    elif action_type_id == 5:
        content = '{\"action\":\"back\"}'
        
    elif action_type_id == 7:
        content = '{\"action\":\"enter\"}'

    elif action_type_id == 11:
        content = '{\"action\":\"task_impossible\"}'
        
    else:
        print('error action')
        print(ann)
        content = 'error'
    return content


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--img_root', type=str, help='Path to the image root directory')
    parser.add_argument('--inp_json_p', type=str, help='Path to the input JSON file')
    parser.add_argument('--out_json_p', type=str, help='Path to the output JSON file')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print_args(args)
    img_root = args.img_root
    inp_json_p = args.inp_json_p
    out_json_p = args.out_json_p

    all_ann = read_json(inp_json_p)
    PROMPT = all_prompts['agent_prompt_for_train']

    all_data = []
    for split, ann_lst in all_ann.items():
        split_data_num = 0
        # one kind of task
        for p_idx, sample in enumerate(ann_lst):
            # one trajectory
            image_lst = []
            action_his = []
            for step_idx in range(len(sample)):
                # one step
                conversation = []
                ann = sample[step_idx]
                instruction = ann['goal']
                img_filename = os.path.join(img_root, f"{ann['img_filename']}.png")

                act_his_str = '\n'.join(action_his[-4:])
                if len(action_his) == 0:
                    act_his_str = 'null'
                    
                prompt = PROMPT.format(instruction=instruction, 
                                       action_history=act_his_str)
                
                content = parse_action_type(ann)
                try:
                    json.loads(content)
                except:
                    print(f'error content: {content}')
                    continue
                action_his.append(content)

                conversation.append({'from': 'human', 'value': prompt})
                conversation.append({'from': 'gpt', 'value': content})
                line = {'conversation': conversation, 'image_lst': [img_filename]}
                all_data.append(line)
                split_data_num += 1
                # break
        #     break
        # break
    print(f'>>> total data num: {len(all_data)}')
    save_json(all_data, out_json_p)


