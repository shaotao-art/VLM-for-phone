import os
import sys
PROJECT_ROOT=os.path.dirname(os.path.abspath('.'))
sys.path.append(PROJECT_ROOT)

from transformers import AutoProcessor
import numpy as np
import logging
from tqdm import tqdm
import pickle 
import argparse


from utils.file_utils import read_json
from utils.helper_utils import smart_resize, print_args


logging.basicConfig(level=logging.INFO)



def get_x_percentile_value(lst, p):
    sorted_lst = sorted(lst)
    index = int(p * (len(sorted_lst) - 1))
    return sorted_lst[index]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--data_p', type=str, help='Path to the data file')
    parser.add_argument('--img_shape_pkl_p', type=str, help='Path to the image shape pickle file')
    parser.add_argument('--data_format', type=str, choices=['custom', 'sharegpt'], help='Format of the data')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--max_img_tokens', type=int, help='Maximum number of image tokens')

    args = parser.parse_args()
    print_args(args)

    data_p = args.data_p
    img_shape_pkl_p = args.img_shape_pkl_p
    data_format = args.data_format
    MODEL_PATH = args.model_path
    max_img_tokens = args.max_img_tokens
    
    assert data_format in ['custom', 'sharegpt']
    min_pixels=4 * 28 * 28
    max_pixels=10000 * 28 * 28 # 包含<|vision_start/end|>
    processor = AutoProcessor.from_pretrained(MODEL_PATH,
                                            min_pixels=min_pixels,
                                            max_pixels=max_pixels,)

    try:
        with open(img_shape_pkl_p, 'rb') as f:
            img_name2shape = pickle.load(f)
    except:
        print(f'always using {max_img_tokens} as the number of image tokens')
        flag = True
    data = read_json(data_p)

    if data_format == 'custom':
        # custom train format
        num_token_lst = []
        for idx, line in enumerate(tqdm(data)):
            # import pdb; pdb.set_trace()
            img_p = line['messages'][0]['content'][0]['image']
            if 'omniact' in data_p:
                img_name = '/'.join(img_p.split('/')[-2:])
            else:
                img_name = os.path.basename(img_p)
                
            prompt = processor.apply_chat_template(line['messages'],
                        tokenize=False,
                        add_generation_prompt=False)

            # only tokenize text 
            # we cal num of img tokens ourself
            inputs = processor(
                text=[prompt],
                padding=False, 
                return_tensors="pt"
            )
            num_text_tokens = inputs.input_ids.size(1)
                    
            if flag:
                single_img_tokens = max_img_tokens
            else:
                h, w = img_name2shape[img_name]
                h_, w_ = smart_resize(height=h, width=w, max_pixels=max_img_tokens * 28 * 28)
                single_img_tokens = h_ / 28 * w_ / 28

 
            num_token_lst.append(num_text_tokens - 1 + single_img_tokens) 
            # -1 for <|image_pad|> in text tokens, which will be replaced by actual img tokens

        p_lst = [i / 10 for i in range(1, 9)]
        for p in p_lst:
            res = get_x_percentile_value(num_token_lst, p)
            print(p, '->', res)

        p_lst = np.linspace(0.9, 1.0, 11)
        for p in p_lst:
            res = get_x_percentile_value(num_token_lst, p)
            print(p.item(), '->', res)

    else:
        token_num_lst = []
        for idx, line in enumerate(tqdm(data)):
            num_tokens = 0
            conversation_lst = line['conversation']
            assert len(conversation_lst) % 2 == 0
            mm_message = []
            for turn_idx in range(len(conversation_lst) // 2):
                inp = conversation_lst[turn_idx * 2]['value']
                oup = conversation_lst[turn_idx * 2 + 1]['value']
                
                imgs = line['image_lst']
                num_img_tokens = 0
                for img_p in imgs:
                    if 'omniact' in data_p:
                        img_name = '/'.join(img_p.split('/')[-2:])
                    else:
                        img_name = os.path.basename(img_p)
                    h, w = img_name2shape[img_name]
                    h_, w_ = smart_resize(height=h, width=w, max_pixels=max_img_tokens * 28 * 28)
                    single_img_tokens = h_ / 28 * w_ / 28
                    num_img_tokens += single_img_tokens
                    if idx == 0:
                        logging.info(f'img shape: {h_}, {w_}')
                        logging.info(f'single img tokens: {single_img_tokens}')
                    
                    
                mm_message += [
                        {"role": "user",
                        "content": [
                            {"type": "text", "text": inp.replace('<image>', '')},
                        ]},
                        {"role": "assistant",
                        "content": [
                            {"type": "text", "text": oup}
                        ]
                        }]
            
            try:
                prompt = processor.apply_chat_template(mm_message,
                                            tokenize=False,
                                            add_generation_prompt=False)

                inputs = processor(
                    text=[prompt],
                    padding=False, 
                    return_tensors="pt"
                )
                num_text_tokens = inputs.input_ids.size(1)
                num_tokens += num_text_tokens + num_img_tokens + 2 
                # +2 for vision start and vison end
                token_num_lst.append(num_tokens)
            except:
                logging.error(f'error in index {idx}')
                continue
            
            if idx == 0:
                logging.info(f'num text tokens: {num_text_tokens}')
                logging.info(f'total token num: {num_tokens}')
                logging.info(processor.tokenizer.decode(inputs.input_ids[0]))

        p_lst = np.linspace(0.9, 1.0, 11)
        for p in p_lst:
            res = get_x_percentile_value(token_num_lst, p)
            print(p, '->', res)




