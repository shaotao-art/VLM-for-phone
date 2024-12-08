import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.img_ops import resize_image_short_side
from utils.file_utils import save_image
from utils.draw_utils import draw_dot, plot_action
from utils.helper_utils import get_action_args
from PIL import Image
import numpy as np
import ast
import logging

logging.basicConfig(level=logging.INFO)


def infer_single_img_qwen(model, processor, img_p, prompt, device):
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img_p,
            },
            {"type": "text", "text": prompt},
            ],
        }
    ]
    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt], 
        images=image_inputs, 
        videos=video_inputs,
        padding=True, 
        return_tensors="pt"
    )
    inputs = inputs.to(device)
    logging.info('>>> text_prompt: ')
    logging.info(text_prompt)
    logging.info('>>> image input: ')
    logging.info(image_inputs)
    logging.info('image token的数量' + str(torch.sum(inputs.input_ids == 151655).item()))
    
    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return output_text
    

if __name__ == '__main__':
    logging.info('可用gpu数量：' + str(torch.cuda.device_count()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_p = "/home/shaotao/PRETRAIN-CKPS/qwen2-vl-2b"
    lora_ckp_p = '/home/shaotao/PROJECTS/VLM_AND_PHONE/training/custom_lora_saves/rep-omniact/checkpoint-936'
    
    
    img_p = './test.jpg'
    instruction = 'open photos'
    action_history = 'null'
    img_size = 336


    img = np.array(Image.open(img_p))
    img = resize_image_short_side(img, img_size)
    img_p = 'tmp.png'
    save_image(img, img_p)

    PROMPT = """Please generate the next move according to the ui screenshot, instruction and previous actions. 
## Instruction: 
{instruction}
 
## Previous actions: 
{action_history}
"""
    prompt = PROMPT.format(instruction=instruction, action_history=action_history)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_p
    ).half().to(device)
    if lora_ckp_p is not None:
        model.load_adapter(lora_ckp_p)
        logging.info('lora ckp loaded')
        
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_p, 
                                              min_pixels=min_pixels, 
                                              max_pixels=max_pixels)
    model_pred = infer_single_img_qwen(model, processor, img_p, prompt, device)
    logging.info(f'model_pred: {model_pred}')
    action, args = get_action_args(model_pred)
    logging.info(f'action: {action}, args: {args}')
    plot_action(img, action, args, out_p='vis.jpg')


