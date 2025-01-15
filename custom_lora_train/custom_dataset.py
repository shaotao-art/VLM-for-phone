from torch.utils.data import Dataset
from data_aug import random_crop_metadata
import random
import json
import os
import sys
from PIL import Image
import numpy as np

from typing import List

sys.path.append('/home/shaotao/PROJECTS/VLM_AND_PHONE/')
from utils.draw_utils import draw_box
from utils.file_utils import save_image


def make_conv_lst(init_prompt: str, 
                inst_lst: List[str], 
                ans_lst: List[str]):
    conv_lst = []
    
    # first turn
    first_user_turn = dict(
        role="user",
        content=[
            dict(type="image"),
            dict(type="text", text=init_prompt),
            dict(type="text", text=inst_lst[0])
        ]
    )
    first_assistant_turn = dict(
        role="assistant",
        content=[
            dict(type="text", text=ans_lst[0])
        ]
    )
    conv_lst.append(first_user_turn)
    conv_lst.append(first_assistant_turn)
    
    # other turns
    for inst, ans in zip(inst_lst[1:], ans_lst[1:]):
        user_turn = dict(
            role="user",
            content=[
                dict(type="text", text=inst)
            ]
        )
        assistant_turn = dict(
            role="assistant",
            content=[
                dict(type="text", text=ans)
            ]
        )
        conv_lst.append(user_turn)
        conv_lst.append(assistant_turn)
    return conv_lst
    


class GroundingDataset(Dataset):
    """ONLY work for ONE image
    expected data format:
    [{
        "img_url": "path/to/img.jpg",
        "element": [
            {
                "instruction": "instruction text",
                "point": [x, y]
            },
            ...
        ]
    }
    ...]
    """
    def __init__(self, 
                 data_path: str,
                 img_root: str,
                 init_prompt: str,
                 pt_format: str,
                 ele_per_img: int,
                 crop_min: float,
                 crop_max: float):
        super().__init__()
        assert pt_format in ['float', 'int']
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.img_root = img_root
        self.init_prompt = init_prompt
        self.point_format = pt_format
        self.element_per_img = ele_per_img # <= 0 means all elements
        self.crop_min = crop_min
        self.crop_max = crop_max
   
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        metadata = self.data[idx]
        img_p = os.path.join(self.img_root, metadata['img_url'])
        img = Image.open(img_p).convert('RGB')
        if self.crop_min != 1.0 or self.crop_max != 1.0:
            img, metadata = random_crop_metadata(img, metadata, (self.crop_min, self.crop_max))
            # img.save(f'cropped_img_{idx}.jpg')

        inst_lst = [_['instruction'] for _ in metadata['element']]
        pt_lst = [_['point'] for _ in metadata['element']]
        # shuffle inst_lst and pt_lst
        shuffle_idx = list(range(len(inst_lst)))
        random.shuffle(shuffle_idx)
        # random sample elements
        if self.element_per_img > 0:
            shuffle_idx = shuffle_idx[:self.element_per_img]
        inst_lst = [inst_lst[i] for i in shuffle_idx]
        pt_lst = [pt_lst[i] for i in shuffle_idx]
        
        if self.point_format == 'float':
            pt_lst = [(round(pt[0], 2), round(pt[1], 2)) for pt in pt_lst]
        elif self.point_format == 'int':
            pt_lst = [(int(pt[0] * 1000), int(pt[1] * 1000)) for pt in pt_lst]
        inst_lst = [f'Instruction: {inst}' for inst in inst_lst]
        pt_lst = [f'[{pt[0]},{pt[1]}]' for pt in pt_lst]

        conv_lst = make_conv_lst(self.init_prompt, inst_lst, pt_lst)
        return dict(input_ids=dict(conv_lst=conv_lst, img=img))


class Loc2FuncDataset(Dataset):
    """ONLY work for ONE image
    expected data format:
    [{
        "img_url": "path/to/img.jpg",
        "element": [
            {
                "instruction": "instruction text",
                "point": [x, y]
            },
            ...
        ]
    }
    ...]
    """
    def __init__(self, 
                 data_path: str,
                 img_root: str,
                 init_prompt: str,
                 pt_format: str,
                 ele_per_img: int,
                 use_som: bool=False,
                 use_ocr: bool=False,
                 crop_min: float=1.0,
                 crop_max: float=1.0):
        super().__init__()
        assert pt_format in ['float', 'int']
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.img_root = img_root
        self.init_prompt = init_prompt
        self.point_format = pt_format
        self.element_per_img = ele_per_img # <= 0 means all elements
        self.use_som = use_som
        self.use_ocr = use_ocr
        # default do not use data augmentation
        self.crop_min = crop_min
        self.crop_max = crop_max
   
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        metadata = self.data[idx]
        img_p = os.path.join(self.img_root, metadata['img_url'])
        img = Image.open(img_p).convert('RGB')
        if self.crop_min != 1.0 or self.crop_max != 1.0:
            img, metadata = random_crop_metadata(img, metadata, (self.crop_min, self.crop_max))

        inst_lst = [_['instruction'] for _ in metadata['element']]
        pt_lst = [_['bbox'] for _ in metadata['element']]
        if self.use_ocr:
            text_lst = [_['text'] for _ in metadata['element']]
        # shuffle inst_lst and pt_lst
        shuffle_idx = list(range(len(inst_lst)))
        random.shuffle(shuffle_idx)
        # random sample elements
        if self.element_per_img > 0:
            shuffle_idx = shuffle_idx[:self.element_per_img]
        inst_lst = [inst_lst[i] for i in shuffle_idx]
        pt_lst = [pt_lst[i] for i in shuffle_idx]
        if self.use_ocr:
            text_lst = [text_lst[i] for i in shuffle_idx]
        
        if self.use_som:
            for pt in pt_lst:
                img = draw_box(np.array(img), pt)
                save_image(img, 'tmp.jpg')
                
                
        if self.point_format == 'float':
            pt_lst = [list(map(lambda x: round(x, 2), pt)) for pt in pt_lst]
        elif self.point_format == 'int':
            pt_lst = [list(map(lambda x: int(x * 1000), pt)) for pt in pt_lst]
        # swap inst_lst and pt_lst to make it compatible with loc2func model
        
        if self.use_ocr:
            pt_lst = [f'Box: ({pt[0]},{pt[1]}),({pt[2]},{pt[3]})\nText: {text}' for pt, text in zip(pt_lst, text_lst)]
        else: 
            pt_lst = [f'Box: ({pt[0]},{pt[1]}),({pt[2]},{pt[3]})' for pt in pt_lst]
        
        conv_lst = make_conv_lst(self.init_prompt, pt_lst, inst_lst)
        return dict(input_ids=dict(conv_lst=conv_lst, img=img))


class AgentDataset(Dataset):
    """Work for multiple images
    expected data format:
    {
        "messages": [
            {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': 'xxxx'}]}, 
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'xxxxxx'}]}
        ]
    }
    """
    def __init__(self, data_path, img_root):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.img_root = img_root
   
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # use input_ids to avoid data be removed by huggingface
        # beacause huggingface trainer will remove keys not in model's forward function
        sample = self.data[idx]
        conv_lst = sample['messages']
        for turn in conv_lst:
            for content in turn['content']:
                if content['type'] == 'image':
                    img_p = os.path.join(self.img_root, content['image'])
                    content['image'] = img_p
        return dict(input_ids=sample)
    

if __name__ == '__main__':
    # test Loc2FuncDataset
    data_path = '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/new_format/tmp.json'
    img_root = '/home/shaotao/DATA/AMEX/screenshot'
    
    # data_path = '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/new_format/gui-act-single-all-new-format.json'
    # img_root = '/home/shaotao/DATA/GUIAct/guiact-single'
    import sys
    sys.path.append('/home/shaotao/PROJECTS/VLM_AND_PHONE/')
    from prompts import ground_prompt
    init_prompt = ground_prompt
    pt_format = 'int'
    ele_per_img = 2
    use_ocr = True
    use_som = True
    crop_min = 1.0
    crop_max = 1.0
    dataset = Loc2FuncDataset(data_path, img_root, init_prompt, pt_format, ele_per_img, use_som, use_ocr, crop_min, crop_max)
    print(len(dataset))
    print(dataset[0])
    
    # # test GroundingDataset
    # from prompts import ground_prompt
    # init_prompt = ground_prompt
    # pt_format = 'int'
    # ele_per_img = 2
    # crop_min = 1.0
    # crop_max = 1.0
    # dataset = GroundingDataset(data_path, img_root, init_prompt, pt_format, ele_per_img, crop_min, crop_max)
    # print(len(dataset))
    # print(dataset[0])
    
    # # test AgentDataset
    # data_path = 'data/agent_train.json'
    # img_root = 'data/images'
    # dataset = AgentDataset(data_path, img_root)
    # print(len(dataset))
    # print(dataset[0])
    
