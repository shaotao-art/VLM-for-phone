from openai import OpenAI
import os
import sys
import cv2
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

from utils.file_utils import (read_image, 
                              read_json, 
                              save_json, 
                              get_image_base64)
from utils.img_ops import resize_image_short_side, smart_resize
from utils.draw_utils import draw_box
from utils.helper_utils import print_args, get_date_str
from prompts import all_prompts


def preprocess_image(img: np.ndarray, 
                     box_int: Tuple[int, int, int, int],
                     use_phone_crop: bool, 
                     vis: bool) -> np.ndarray:
    """- whether to use phone crop or not
    - whether to visualize the box or not"""
    if use_phone_crop:
        # crop the image based on the box to make a phone size image
        h, w = img.shape[:2]
        box_in_float = [_ / 1000 for _ in box_int]
        box_in_pixel = box_in_float[0] * w, box_in_float[1] * h, box_in_float[2] * w, box_in_float[3] * h
        cent_x = (box_in_float[0] + box_in_float[2]) / 2
        cent_x = int(w * cent_x)
        x1 = max(cent_x - h // 4, 0)
        x2 = x1 + h // 2
        img = img[:, x1:x2, :]
        
        new_b_x1 = box_in_pixel[0] - x1
        new_b_x2 = box_in_pixel[2] - x1
        new_box = [new_b_x1, box_in_pixel[1], new_b_x2, box_in_pixel[3]]
        patch_h, patch_w = img.shape[:2]
        x1, y1, x2, y2 = new_box[0] / patch_w, new_box[1] / patch_h, new_box[2] / patch_w, new_box[3] / patch_h
        if vis == True:
            img = draw_box(img, (x1, y1, x2, y2), color=(255, 0, 0))
    else:
        box_in_float = [_ / 1000 for _ in box_int] 
        if vis == True:
            img = draw_box(img, box_in_float, color=(255, 0, 0))
    return img

def infer(client, 
          model_name,
          prompt: str, 
          img_p: str, 
          box_int: Tuple[int, int, int, int], 
          use_phone_crop: bool, 
          vis: bool, 
          temprature: float):
    image_np = read_image(img_p)
    ori_img_shape = image_np.shape[:2]
    h, w = ori_img_shape
    image_np = preprocess_image(image_np, box_int, use_phone_crop, vis)
    if use_smart_resize:
        h_bar, w_bar = smart_resize(height=h, width=w, max_pixels=14 * 14 * 4 * max_img_tokens)
        image_np = cv2.resize(image_np, (w_bar, h_bar))
    else:
        if h > w:
            # phone
            image_np = resize_image_short_side(image_np, phone_img_short_side_size)
        else:
            # pad
            image_np = resize_image_short_side(image_np, pad_img_short_side_size)
        
        
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": get_image_base64(image_np),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        temperature=temprature,
        seed=42
    )
    model_pred = chat_response.choices[0].message.content
    return dict(pred=model_pred, img_shape=ori_img_shape)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on images using OpenAI model.")
    
    parser.add_argument('--openai_api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--openai_api_base', type=str, required=True, help='OpenAI API base URL')
    parser.add_argument('--temprature', type=float, required=True, help='Temperature for the model')
    parser.add_argument('--num_thread', type=int, default=20, help='Number of threads to use for inference')

    parser.add_argument('--model_name', type=str, required=True, help='Model name to use for inference')
    parser.add_argument('--prompt_type', type=str, required=True, help='prompt for the model')
    parser.add_argument('--phone_img_short_side_size', type=int, default=-1, help='Short side size for phone images')
    parser.add_argument('--pad_img_short_side_size', type=int, default=-1, help='Short side size for pad images')
    parser.add_argument('--use_smart_resize', action='store_true', help='Use smart resize for images')
    parser.add_argument('--inp_json_p', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--out_json_p', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--img_root', type=str, required=True, help='Root directory for images')
    parser.add_argument('--max_img_tokens', type=int, help='Maximum number of image tokens in test')
    parser.add_argument('--use_phone_crop', action='store_true', help='Use phone crop for images')
    parser.add_argument('--vis', action='store_true', help='Visualize the crop')
    return parser.parse_args()

    
if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    
    openai_api_key = args.openai_api_key
    openai_api_base = args.openai_api_base
    model_name = args.model_name
    phone_img_short_side_size = args.phone_img_short_side_size
    pad_img_short_side_size = args.pad_img_short_side_size
    use_smart_resize = args.use_smart_resize
    inp_json_p = args.inp_json_p
    out_json_p = args.out_json_p
    img_root = args.img_root
    prompt_type = args.prompt_type
    temprature = args.temprature
    num_thread = args.num_thread
    max_img_tokens = args.max_img_tokens
    use_phone_crop = args.use_phone_crop
    vis = args.vis
    if use_smart_resize:
        assert phone_img_short_side_size == -1, "Cannot use both smart resize and fixed short side size"
        assert pad_img_short_side_size == -1, "Cannot use both smart resize and fixed short side size"
    
    
    PROMPT = all_prompts[prompt_type].replace('<image>', ' ')
    day_str = get_date_str()
    out_root = os.path.join('out', day_str, model_name)
    os.makedirs(out_root, exist_ok=True)
    out_json_p = os.path.join(out_root, args.out_json_p)
    if os.path.exists(out_json_p):
        print(f"Output file already exists: {out_json_p}")
        sys.exit(0)
    
    client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
    

    data = read_json(inp_json_p)
    # preprocess all the box
    int_box_lst = []
    for d_idx in range(len(data)):
        box = data[d_idx]['box']
        int_box = [int(_ * 1000) for _ in box]
        int_box_lst.append(int_box)
    assert len(int_box_lst) == len(data)
    
    # pack all params into list
    params = [(client, 
                model_name,
                PROMPT.format(x1=int_box_lst[d_idx][0], 
                              y1=int_box_lst[d_idx][1], 
                              x2=int_box_lst[d_idx][2], 
                              y2=int_box_lst[d_idx][3]) \
                    if 'ocr' not in prompt_type else \
                        PROMPT.format(x1=int_box_lst[d_idx][0], 
                                      y1=int_box_lst[d_idx][1], 
                                      x2=int_box_lst[d_idx][2], 
                                      y2=int_box_lst[d_idx][3], 
                                      text=data[d_idx]['text']),
                os.path.join(img_root,
                            data[d_idx]['img_name']),
                int_box_lst[d_idx],
                use_phone_crop,
                vis,
                temprature) for d_idx in range(len(data))]

    # params = params[:200]
        
    print('sample prompt:', params[0][2])
    print('sample pred: ', infer(*params[0])['pred'])
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        # use map to keep the order of results
        results = list(tqdm(executor.map(lambda param: infer(*param), 
                                         params),
                            total=len(data)))

        
    out = []
    for d_idx, model_pred in enumerate(results):
        out_line = deepcopy(data[d_idx])
        out_line['model_pred'] = model_pred['pred']
        out_line['ori_img_shape'] = model_pred['img_shape']
        out.append(out_line)
    save_json(out, out_json_p)