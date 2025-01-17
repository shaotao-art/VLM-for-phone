from openai import OpenAI
import os
import cv2
import logging


from utils.file_utils import read_image, save_image, get_image_base64, read_json
from utils.file_utils import read_image, get_image_base64
from utils.draw_utils import draw_dot
from utils.img_ops import smart_resize_img

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    openai_api_key = "shaotao"
    openai_api_base = "http://localhost:8005/v1"
    # model_name = 'qwen2_VL_7B'
    model_name = 'tt'
    max_img_tokens = 1280
    temprature = 0.3
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    box = [89, 3, 110, 22]
    prompt = """<image>Output the location of target element according to the given instruction.
## Instruction
{instruction}"""
    prompt = prompt.format(instruction="view more options")

    
    img = read_image("/home/shaotao/PROJECTS/VLM_AND_PHONE/tmp.jpeg")
    img = smart_resize_img(img, max_img_tokens * 28 * 28)
    

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
    x, y = eval(model_pred)
    x = x / 1000
    y = y / 1000
    img = draw_dot(img, (x, y), color=(0, 255, 0), radius=20)
    save_image(img, 'vis.jpg')
    
    
