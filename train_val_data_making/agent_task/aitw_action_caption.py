"""aitw action caption dataset making"""
import json
import sys
import os
import argparse
from typing import Tuple, Dict, List
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.file_utils import read_image, read_json, save_image, save_json
from utils.helper_utils import print_args
from prompts import all_prompts

def format_his_info(action_his:List[Dict], use_cot_his: bool):
    act_his_str = ''
    for his in action_his:
        his = json.loads(his)
        if 'action' in his and use_cot_his:
            act_his_str += f"action: {his['action']} "
        if 'action_value' not in his:
            act_his_str += f"action_type: {his['action_type']}\n"
        else:
            act_his_str += f"action_type: {his['action_type']}, action_value: {his['action_value']}\n"
    return act_his_str.strip()


def determine_swipe_direction(str_pt: Tuple[float, float], 
                              end_pt: Tuple[float, float]) -> str:
    delta_x = end_pt[0] - str_pt[0]
    delta_y = end_pt[1] - str_pt[1]

    if abs(delta_x) > abs(delta_y):
        if delta_x > 0:
            return 'left'
        else:
            return 'right'
    else:
        if delta_y > 0:
            return 'up'
        else:
            return 'down'

def parse_action_type(ann, cot_ann: Dict):
    action_type_id = int(ann['action_type_id'])
    action_type_text = ann["action_type_text"]
    touch_pt = ann['touch']
    lift_pt = ann['lift']
    text = ann['type_text']
    # 3: type
    # 4: click
    # 5: back
    # 6: home
    # 7: enter
    # 10: task complete
    # 11: task impossible

    
    if action_type_id == 4:
        if action_type_text == 'click':
            cent_x = (touch_pt[0] + lift_pt[0]) / 2
            cent_y = (touch_pt[1] + lift_pt[1]) / 2
            cent_x = int(cent_x * 1000)
            cent_y = int(cent_y * 1000) 
            answer = dict(
                action_type='click',
                action_value=(cent_x, cent_y)
            )

        elif action_type_text in ['scroll up', 'scroll down', 'scroll left', 'scroll right']:
            direction = determine_swipe_direction(touch_pt, lift_pt)
            answer = dict(
                action_type=f'scroll_{direction}',
            )
        else:
            print('error click')
            print(ann)
            
        
        
    elif action_type_id == 3:
        answer = dict(
            action_type='type',
            action_value=text
        )
        
    elif action_type_id == 10:
        answer = dict(
            action_type='task_complete'
        )
        
    elif action_type_id == 6:
        answer = dict(
            action_type='home'
        )

    elif action_type_id == 5:
        answer = dict(
            action_type='back'
        )
        
    elif action_type_id == 7:
        answer = dict(
            action_type='enter'
        )

    elif action_type_id == 11:
        answer = dict(
            action_type='task_impossible'
        )
        
    else:
        print('error action')
        print(ann)
        
    if cot_ann is not None:
        cot_ann.update(answer)
        return json.dumps(cot_ann)
    else:
        return json.dumps(answer)


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--img_root', type=str, help='Path to the image root directory')
    parser.add_argument('--inp_json_p', type=str, help='Path to the input JSON file')
    parser.add_argument('--out_json_p', type=str, help='Path to the output JSON file')
    parser.add_argument('--hist_len', type=int, help='Length of the action history')
    parser.add_argument('--cot_ann_p', type=str, default=None, help='multi level annotation json path')
    parser.add_argument('--cot_level', type=str, default=None, help='cot annotation level')
    parser.add_argument('--use_cot_his', action='store_true', help='whether to include action description in history')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print_args(args)
    img_root = args.img_root
    inp_json_p = args.inp_json_p
    out_json_p = args.out_json_p
    hist_len = args.hist_len
    cot_ann_p = args.cot_ann_p
    cot_level = args.cot_level
    use_cot_his = args.use_cot_his
   
    if cot_level is not None:
        assert cot_level in ['action', 'thoughts']

    all_ann = read_json(inp_json_p)
    PROMPT = all_prompts['agent_action_caption']
    if cot_ann_p is not None:
        cot_ann = pd.read_excel(cot_ann_p)
        cot_ann.set_index('img_name', inplace=True)

    all_data = []
    for split, ann_lst in all_ann.items():
        # one kind of task
        for p_idx, sample in enumerate(tqdm(ann_lst)):
            # one trajectory
            image_lst = []
            action_his = []
            for step_idx in range(len(sample)):
                # one step
                conversation = []
                ann = sample[step_idx]
                instruction = ann['goal']
                img_filename = f"{ann['img_filename']}.png"
                img_p = os.path.join(img_root, img_filename)
                # img = read_image(img_p)
                # save_image(img, f'{step_idx}.png')
                
                cot = None
                if cot_ann_p is not None:
                    dir_name = os.path.dirname(img_filename)
                    filename = os.path.basename(img_filename)
                    df_key = f"{dir_name}_{filename.replace('.png', '.jpg')}"
                    try:
                        multi_level_ann_line = cot_ann.loc[df_key]
                    except:
                        print('one img do not have multi-level ann, skipping...')
                        continue
                    # previous_actions = multi_level_ann_line['previous_actions']
                    observation = multi_level_ann_line['observation']
                    thought = multi_level_ann_line['thought']
                    action = multi_level_ann_line['action']
                    if cot_level == 'action':
                        cot = dict(
                            action=action
                        )
                    else:
                        cot = dict(
                            observation=observation,
                            thought=thought,
                            action=action
                        )
                # history format: action(optional), action_type, action_params   
                act_his_str = format_his_info(action_his[-hist_len:], use_cot_his)
                if len(action_his) == 0:
                    act_his_str = 'null'
                
                answer = parse_action_type(ann, cot)
                action_his.append(answer)
                answer = json.loads(answer)
                prompt = PROMPT.format(instruction=instruction, 
                                       action_history=act_his_str,
                                       action_type=answer['action_type'],
                                       action_value=answer.get('action_value', 'null'))
                if '<image>' not in prompt:
                    prompt = '<image>' + prompt
                    
                if 'test' not in inp_json_p:
                    gt = answer['action']

                conversation.append({'from': 'human', 'value': prompt})
                if 'test' not in inp_json_p:
                    conversation.append({'from': 'gpt', 'value': gt})
                line = {'conversation': conversation, 'image_lst': [img_filename]}
                all_data.append(line)
                print('>>>>> sample')
                print(line['conversation'][0]['value'])
                print(line['conversation'][1]['value'])
                # break
            break
        break
    print(f'>>> total data num: {len(all_data)}')
    # save_json(all_data, out_json_p)