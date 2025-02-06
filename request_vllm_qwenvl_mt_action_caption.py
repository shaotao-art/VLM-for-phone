import os
import sys
from openai import OpenAI
import cv2
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import argparse

from utils.file_utils import (read_image, 
                              read_json, 
                              save_json, 
                              get_image_base64)
from utils.img_ops import resize_image_short_side
from utils.helper_utils import smart_resize, print_args, get_date_str



def infer(client, model_name, prompt, img_p, temprature):
    image_np = read_image(img_p)
    if use_smart_resize:
        h, w = image_np.shape[:2]
        h, w = smart_resize(h, w, max_pixels=max_img_tokens * 28 * 28)
        image_np = cv2.resize(image_np, (w, h))
    else:
        image_np = resize_image_short_side(image_np, img_short_side_size)
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
    finish_reason = chat_response.choices[0].finish_reason
    return dict(pred=model_pred, finish_reason=finish_reason)



def get_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--api_key', type=str, help='OpenAI API key')
    parser.add_argument('--api_base', type=str, help='OpenAI API base URL')
    parser.add_argument('--temprature', type=float, help='Temperature')
    parser.add_argument('--num_thread', type=int, default=20, help='Number of threads')
    parser.add_argument('--img_root', type=str, required=True, help='Image root path')
    
    
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--inp_json_p', type=str, required=True, help='Input JSON path')
    parser.add_argument('--out_json_p', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--img_short_side_size', type=int, default=-1, help='Image short side size')
    parser.add_argument('--max_img_tokens', type=int, required=True, help='Maximum number of image tokens')
    parser.add_argument('--use_smart_resize', action='store_true', help='Use smart resize for images')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print_args(args)
    openai_api_key = args.api_key
    openai_api_base = args.api_base
    model_name = args.model_name
    inp_json_p = args.inp_json_p
    out_json_p = args.out_json_p
    img_short_side_size = args.img_short_side_size
    temprature = args.temprature
    num_thread = args.num_thread
    max_img_tokens = args.max_img_tokens
    use_smart_resize = args.use_smart_resize
    img_root = args.img_root
    
    if use_smart_resize:
        assert img_short_side_size == -1, "Cannot use both smart resize and fixed short side size"
    
    client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
    data = read_json(inp_json_p)
    day_str = get_date_str()
    out_root = os.path.join('out', day_str, model_name)
    os.makedirs(out_root, exist_ok=True)
    out_json_p = os.path.join(out_root, args.out_json_p)
    if os.path.exists(out_json_p):
        print(f"Output file already exists: {out_json_p}")
        sys.exit(0)
    
    
    # pack all params into list
    params = [(client, 
                model_name,
                data[d_idx]['conversation'][0]['value'].replace('<image>', ''), 
                os.path.join(img_root, data[d_idx]['image_lst'][0]),
                temprature) for d_idx in range(len(data))]

    # params = params[:20]
    print('sample prompt: ', params[2][2])
    model_pred = infer(*params[2])
    print('sample prediction: ', model_pred)
    
    
    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        # use map to keep the order of results
        results = list(tqdm(executor.map(lambda params: infer(*params), 
                                         params),
                            total=len(params)))

        
    out = []
    for d_idx, model_pred in enumerate(results):
        out_line = deepcopy(data[d_idx])
        out.append(out_line)
        out[d_idx]['conversation'].append({'from': 'prediction', 
                                           'value': model_pred})
    save_json(out, out_json_p)

