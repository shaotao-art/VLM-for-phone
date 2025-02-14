"""expect data format:
[
    {
        "img_name": "macos_icons/1.png",
        "bbox": [0.1, 0.2, 0.3, 0.4],
        "text": "some text",
    },
    ...
]
"""
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
                              get_image_base64,
                              save_image)
from utils.img_ops import resize_image_short_side, smart_resize
from utils.draw_utils import draw_box
from utils.helper_utils import print_args, get_date_str
from prompts import all_prompts


def cut_image_by_box(image: np.ndarray, 
                     box: Tuple[float, float, float, float], 
                     margin: float = 0.1) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """crop img according to the box"""
    height, width = image.shape[:2]
    
    x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(box, [width, height, width, height])]
    
    margin_x = int(margin * width)
    margin_y = int(margin * height)
    
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    quad_x = int(center_x > width / 2)
    quad_y = int(center_y > height / 2)
    
    quad_boundaries = {
        (0, 0): ((-margin_x, width // 2 + margin_x), (-margin_y, height // 2 + margin_y)),
        (1, 0): ((width // 2 - margin_x, width + margin_x), (-margin_y, height // 2 + margin_y)),
        (0, 1): ((-margin_x, width // 2 + margin_x), (height // 2 - margin_y, height + margin_y)),
        (1, 1): ((width // 2 - margin_x, width + margin_x), (height // 2 - margin_y, height + margin_y))
    }
    
    quad_x_start, quad_x_end = quad_boundaries[(quad_x, quad_y)][0]
    quad_y_start, quad_y_end = quad_boundaries[(quad_x, quad_y)][1]
    
    quad_x_start = min(quad_x_start, x1 - margin_x)
    quad_x_end = max(quad_x_end, x2 + margin_x)
    quad_y_start = min(quad_y_start, y1 - margin_y)
    quad_y_end = max(quad_y_end, y2 + margin_y)
    
    quad_x_start, quad_x_end = max(0, quad_x_start), min(width, quad_x_end)
    quad_y_start, quad_y_end = max(0, quad_y_start), min(height, quad_y_end)
    
    cut_image = image[quad_y_start:quad_y_end, quad_x_start:quad_x_end]
    
    cut_width = quad_x_end - quad_x_start
    cut_height = quad_y_end - quad_y_start
    
    adjusted_x1 = max(0, x1 - quad_x_start) / cut_width
    adjusted_y1 = max(0, y1 - quad_y_start) / cut_height
    adjusted_x2 = min(x2 - quad_x_start, cut_width) / cut_width
    adjusted_y2 = min(y2 - quad_y_start, cut_height) / cut_height
    
    adjusted_box = (adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2)
    
    return cut_image, adjusted_box


def infer(client, 
          model_name,
          line,
          vis: bool, 
          temprature: float):
    try:
        img_filename = line['img_name']
        box = line['bbox']
        img_p = os.path.join(img_root, img_filename)
        image_np = read_image(img_p)
        if use_crop: # use crop for all os systems
            image_np, box = cut_image_by_box(image_np, box)
        h, w = image_np.shape[:2]
        if vis:
            image_np = draw_box(image_np, box, thickness=5, color=(255, 0, 0))
        
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

        int_box_lst = [int(_ * 1000) for _ in box]
        if 'ocr' not in prompt_type:
            prompt = PROMPT.format(x1=int_box_lst[0], 
                            y1=int_box_lst[1], 
                            x2=int_box_lst[2], 
                            y2=int_box_lst[3]) 
        else:
            prompt = PROMPT.format(x1=int_box_lst[0], 
                                y1=int_box_lst[1], 
                                x2=int_box_lst[2], 
                                y2=int_box_lst[3], 
                                text=line.get('text', 'null'))
            
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
        return dict(pred=model_pred, img_shape=(h, w))
    except Exception as e:
        print(f"Error: {e}")
        return dict(pred='Error', img_shape=(-1, -1))

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
    parser.add_argument('--use_crop', action='store_true', help='Use croped images for high resolution imgs')
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
    use_crop = args.use_crop
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
    # pack all params into list
    params = [(client, 
                model_name,
                data[d_idx],
                vis,
                temprature) for d_idx in range(len(data))]

    # params = params[:200]
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        # use map to keep the order of results
        results = list(tqdm(executor.map(lambda param: infer(*param), 
                                         params),
                            total=len(params)))

        
    out = []
    for d_idx, model_pred in enumerate(results):
        out_line = deepcopy(data[d_idx])
        out_line['model_pred'] = model_pred['pred']
        out_line['ori_img_shape'] = model_pred['img_shape']
        out.append(out_line)
    save_json(out, out_json_p)