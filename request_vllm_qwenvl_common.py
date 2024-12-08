from openai import OpenAI
import base64
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import time
from colorama import Fore, Style, init
import logging
import ast

from utils.file_utils import read_image, save_image, get_image_base64
from utils.img_ops import resize_image_short_side
from utils.draw_utils import draw_grid
from utils.helper_utils import get_action_args

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    openai_api_key = "shaotao"
    openai_api_base = "http://localhost:8001/v1"
    model_name = '7b-pt2func'

    

    img_short_side_size = 448
    img_p = 'TEST_IMGS/2_SCROLL.png'
    temprature = 0.3
    # prompt = 'OCR this image'
    prompt = "detect the bounding box of '什么样的代码一看就知道是ai写的'"
    logging.info(">>> Prompt: \n" + prompt)

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    img = read_image(img_p)
    img = resize_image_short_side(img, img_short_side_size)
    
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
                            "url": get_image_base64(img),
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
    logging.info(">>> finish reason: \n" + chat_response.choices[0].finish_reason)
    logging.info(">>> Chat response: \n" + model_pred)
    
