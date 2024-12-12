import json
import numpy as np
import os
from tqdm import tqdm
import sys
import argparse
import json
sys.path.append(os.path.dirname(os.path.abspath('.')))


from utils.file_utils import read_json, save_json, read_pkl
from utils.helper_utils import float2_0_1000, print_args
from prompts import all_prompts



def parse_args():
    parser = argparse.ArgumentParser(description="Process mind2web data")
    parser.add_argument('--img_root', type=str, required=True, help='Root directory of images')
    parser.add_argument('--inp_json_p', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--out_json_p', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--img_shape_pkl_p', type=str, required=True, help='Image shapes pickle file path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    img_root = args.img_root
    inp_json_p = args.inp_json_p
    out_json_p = args.out_json_p
    img_shape_pkl_p = args.img_shape_pkl_p
    
    
    all_ann = read_json(inp_json_p)
    img_shapes = read_pkl(img_shape_pkl_p)
    PROMPT = all_prompts['agent_prompt_for_train']
    
    mind2webaction2action = dict(
        CLICK='click',
        HOVER='click',
        ENTER='click',
        SELECT='select',
        TYPE='type'
    )    

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
                print('one file do not exist')
                continue
            
            img_height, img_width = img_shapes[filename]
            
            text = ann["operation"]["value"].replace('"', "'")
            # cent_point
            action_type = ann["operation"]["original_op"]
            point_x = ann["bbox"]["x"] + (ann["bbox"]["width"] / 2)
            point_y = ann["bbox"]["y"] + (ann["bbox"]["height"] / 2)
            point_x = point_x / img_width
            point_y = point_y / img_height
            try:
                pt = list(map(float2_0_1000, (point_x, point_y)))
            except:
                print(f'error point: {point_x, point_y}')
                continue
            
            if 'test' in inp_json_p:
                # make gt
                box = [ann["bbox"]["x"], ann["bbox"]["y"], ann["bbox"]["width"], ann["bbox"]["height"]]
                box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                box = box[0] / img_width, box[1] / img_height, box[2] / img_width, box[3] / img_height
                try:
                    box = list(map(float2_0_1000, box))
                except:
                    print(f'error box: {box}')
                    continue
            
            act_his_str = '\n'.join(action_his[-4:])
            if len(action_his) == 0:
                act_his_str = 'null'
                
            prompt = PROMPT.format(instruction=instruction, action_history=act_his_str)
            if action_type in ['CLICK', 'HOVER', 'ENTER']:
                content = f'{{\"action\":\"click\",\"pt\":[{pt[0]},{pt[1]}]}}'
                action_his.append(content) 
            elif action_type == 'SELECT':
                content = f'{{\"action\":\"select\",\"pt\":[{pt[0]},{pt[1]}],\"text\":\"{text}\"}}'
                action_his.append(content)  
            elif action_type == 'TYPE':
                content = f'{{\"action\":\"type\",\"pt\":[{pt[0]},{pt[1]}],\"text\":\"{text}\"}}'
                action_his.append(content)
            else:
                raise ValueError(f'Unknown action type: {action_type}')
            try:
                json.loads(content)
            except:
                print(f'error content: {content}')
                continue
            
            conversation.append({'from': 'human', 'value': prompt})
            conversation.append({'from': 'gpt', 'value': content})
            
            # for test data add gt for evaluation
            if 'test' in inp_json_p:
                gt = dict(
                    action_type=mind2webaction2action[action_type],
                    box=box,
                    text=text,
                    ann_id=annot_id
                )
                conversation.append({'from': 'gt', 'value': gt})
            
            line = {'conversation': conversation, 'image_lst': [img_filename]}
            all_data.append(line)
            if len(all_data) == 1:
                print(f'>>> sample: {line}')

    print(f'>>> total data num: {len(all_data)}')
    save_json(all_data, out_json_p)





