import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath('..'))
sys.path.append(PROJECT_ROOT)

from openai import OpenAI
import os
import cv2
from tqdm import tqdm
from copy import deepcopy
import argparse
from concurrent.futures import ThreadPoolExecutor


from utils.file_utils import (read_image, 
                              read_json, 
                              save_json, 
                              get_image_base64)
from utils.img_ops import resize_image_short_side
from utils.helper_utils import smart_resize
from prompts import all_prompts



def infer(client, model_name, prompt, img_p, temprature):
    image_np = read_image(img_p)
    ori_img_shape = image_np.shape[:2]
    h, w = ori_img_shape
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
    parser.add_argument('--temprature', type=float, default=0.3, help='Temperature for the model')
    parser.add_argument('--num_thread', type=int, default=20, help='Number of threads to use for inference')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with limited samples')

    parser.add_argument('--model_name', type=str, required=True, help='Model name to use for inference')
    parser.add_argument('--prompt_type', type=str, required=True, help='prompt for the model')
    parser.add_argument('--phone_img_short_side_size', type=int, default=-1, help='Short side size for phone images')
    parser.add_argument('--pad_img_short_side_size', type=int, default=-1, help='Short side size for pad images')
    parser.add_argument('--use_smart_resize', action='store_true', help='Use smart resize for images')
    parser.add_argument('--inp_json_p', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--out_json_p', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--img_root', type=str, required=True, help='Root directory for images')
    parser.add_argument('--max_img_tokens', type=int, help='Maximum number of image tokens in test')
    return parser.parse_args()

    
if __name__ == '__main__':
    args = parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    openai_api_key = args.openai_api_key
    openai_api_base = args.openai_api_base
    model_name = args.model_name
    phone_img_short_side_size = args.phone_img_short_side_size
    pad_img_short_side_size = args.pad_img_short_side_size
    use_smart_resize = args.use_smart_resize
    inp_json_p = args.inp_json_p
    out_json_p = args.out_json_p
    img_root = args.img_root
    debug = args.debug
    prompt_type = args.prompt_type
    temprature = args.temprature
    num_thread = args.num_thread
    max_img_tokens = args.max_img_tokens
    
    PROMPT = all_prompts[prompt_type]
    
    if 'mobile' in inp_json_p:
        task = 'mobile'
    elif 'desktop' in inp_json_p:
        task = 'desktop'
    elif 'web' in inp_json_p:
        task = 'web'
    else:
        raise ValueError(f'Unknown task for input JSON file: {inp_json_p}')
            
    if use_smart_resize:
        assert phone_img_short_side_size == -1, "Cannot use both smart resize and fixed short side size"
        assert pad_img_short_side_size == -1, "Cannot use both smart resize and fixed short side size"
    
    os.makedirs('output_datas', exist_ok=True)
    out_json_p = f'out/{out_json_p}'
    
    client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
    

    data = read_json(inp_json_p)
    prompt = PROMPT.format(instruction=data[0]['instruction'])
    img_p = os.path.join(img_root, data[0]['img_filename'])
    model_pred_ = infer(client, model_name, prompt, img_p, temprature)
    print(">>> sample data: \n" + str(data[0]))
    print(">>> sample img_p: \n" + img_p)
    print(">>> sample prompt: \n" + prompt)
    print(">>> sample model_pred: \n" + model_pred_['pred'])
    
    # pack all params into list
    params = [(client, 
                model_name,
                PROMPT.format(instruction=data[d_idx]['instruction']), 
                os.path.join(img_root,
                            data[d_idx]['img_filename']),
                temprature) for d_idx in range(len(data))]
    if debug:
        params = params[:20]
    
    print(f'len samples: {len(params)}')
    print(f'file out: {out_json_p}')
    
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
    if not debug:
        save_json(out, out_json_p)