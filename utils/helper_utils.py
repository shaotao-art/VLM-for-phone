import sys
sys.path.append('/home/shaotao/PROJECTS/VLM_AND_PHONE')

from typing import List, Tuple
import math
import logging
import torch
import cv2
from datetime import date
from openai import OpenAI


from utils.file_utils import read_image, get_image_base64



def get_date_str():
    today = date.today()
    formatted_date = today.strftime("%m-%d")
    return formatted_date

def float2_0_1000(x: float) -> str:
    if x == 1.0:
        x = 0.999
    assert x < 1.0 and x >= 0.0, f'get input: {x}'
    return f'{int(x * 1000)}'


def print_args(args):
    print("Input args:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")


def read_and_print_args(file_path):
    args = torch.load(file_path)
    print_args(args)

def smart_resize(height: int, 
                 width: int, 
                 max_pixels: int,
                 factor: int = 28, 
                 min_pixels: int = 56 * 56, 
                 ):

    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def get_num_img_tokens(height: int, width: int, factor: int = 28):
    h_bar, w_bar = smart_resize(height, width, factor)
    return h_bar / factor * w_bar / factor



def request_vllm(model_name, prompt, img, temprature, num_img_tokens, openai_api_key, openai_api_base):
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    h, w = img.shape[:2]
    h, w = smart_resize(h, w, max_pixels=num_img_tokens * 28 * 28)
    image_np = cv2.resize(img, (w, h))

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
    return model_pred
