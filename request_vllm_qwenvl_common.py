from openai import OpenAI
import os
import cv2
import logging


from utils.file_utils import read_image, save_image, get_image_base64, read_json
from utils.file_utils import read_image, get_image_base64
from utils.draw_utils import draw_dot, draw_box
from utils.img_ops import smart_resize_img
from prompts import all_prompts

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    openai_api_key = "shaotao"
    openai_api_base = "http://localhost:8003/v1"
    # model_name = 'qwen2-vl-7b'
    model_name = 'guiact-box2func-som'
    prompt_type = 'box2func_with_som_test'
    # prompt_type = 'box2func_test'
    max_img_tokens = 1344
    temprature = 0.0
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    box_float = [
            0.11221048235893201,
            0.9452250599861141,
            0.13958910107612602,
            0.9910423159599301
        ]
    box = [int(x * 1000) for x in box_float]
    prompt = all_prompts[prompt_type]
    prompt = prompt.format(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
    
    
    img = read_image("/home/shaotao/DATA/os-altas/linux-mac-merged/20240905_143704_screenshot.png")
    if 'som' in prompt_type:
        img = draw_box(img, box_float, color=(255, 0, 0))
    img = smart_resize_img(img, max_img_tokens * 28 * 28)
    save_image(img, 'test.jpg')
    

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
    logging.info(">>> prompt: \n" + prompt)
    logging.info(">>> finish reason: \n" + chat_response.choices[0].finish_reason)
    logging.info(">>> Chat response: \n" + model_pred)
    
    
    
