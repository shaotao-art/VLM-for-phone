from typing import List, Tuple
import math
import logging
import torch


def get_action_args(x: str) -> Tuple[str, str]:
    try:
        str_idx = x.index('(')
        # find the last ')'
        end_idx = x.rfind(')')
        return x[:str_idx], x[str_idx+1:end_idx]
    except:
        logging.error(f'Error in split action and args: {x}')
        return 'ERROR', 'ERROR'

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
