"""out data format:
[
    {
        "img_url": "img_path",
        "element": [
            {
                "instruction": "click",
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "point": [0.2, 0.3],
                "text": "text"
            },
            ...
        ]
    },
    ...
]
"""
import os
import sys
PROJECT_ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
import pandas as pd
from tqdm import tqdm

from utils.file_utils import read_image, read_json, save_image, save_json
from utils.draw_utils import draw_box
from utils.helper_utils import float2_0_1000
from prompts import ground_prompt_for_train, ground_prompt_continue

if __name__ == '__main__':
    data_p = '/home/shaotao/DATA/omniact-SHOWUI-8k/metadata/hf_train.json'
    img_root = '/home/shaotao/DATA/omniact-SHOWUI-8k/screenshots'
    out_json_p = 'omniact-100-train.json'
    data = read_json(data_p)
    data = pd.DataFrame(data)
    
    init_prompt = ground_prompt_for_train
    continue_prompt = ground_prompt_continue

    all_datas = []
    for idx in tqdm(range(len(data))):
        row = data.iloc[idx]
        img_filename = row['img_url']
        # get image path
        # import pdb; pdb.set_trace()
        img_p = os.path.join(img_root, img_filename)
        if idx % 50 == 0:
            img = read_image(img_p)
        # shuffle all elements
        all_elements = row['element']
        conversation_lst = []
        for ele in all_elements:
            # get input prompt
            instructions = ele['instruction']
            if len(conversation_lst) == 0:
                prompt = init_prompt.format(instruction=instructions)
            else:
                prompt = continue_prompt.format(instruction=instructions)
            # get answer
            x1, y1, x2, y2 = ele['bbox']
            pt = ((x1 + x2) / 2, (y1 + y2) / 2)
            pt = list(map(float2_0_1000, pt))
            ans = f'({pt[0]},{pt[1]})'
            conversation_lst.append({'from': 'human', 'value': prompt})
            conversation_lst.append({'from': 'gpt', 'value': ans})
            if idx % 50 == 0:
                img = draw_box(img, (x1, y1, x2, y2))
        if idx % 50 == 0:
            save_image(img, f'tmp_{idx}.jpg')
        # break
        line = {'conversation': conversation_lst, 'image_lst': [img_p]}
        all_datas.append(line)
    print('total data num: ', len(all_datas))
    save_json(all_datas, out_json_p)