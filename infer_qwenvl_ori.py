import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.img_ops import resize_image_short_side
from utils.file_utils import save_image
from utils.draw_utils import draw_dot
from PIL import Image
import numpy as np
import ast


def infer_single_img_qwen(model, processor, img_p, prompt, deivce):
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
    print('>>> text_prompt: ')
    print(text_prompt)
    print('>>> image input: ')
    print(image_inputs)
    print('image token的数量', torch.sum(inputs.input_ids == 151655).item())
    
    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return output_text
    

if __name__ == '__main__':
    print('可用gpu数量：', torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_p = "/home/dmt/shao-tao-working-dir/ALL-PRETRAIN-LLM-CKPS/qwen2_VL_7B"
    img_p = '/home/dmt/shao-tao-working-dir/LLM&PHONE/tmp/nature_img.jpg'
    # prompt = "提取图像中的所有的加粗的文本"
    prompt = 'detect the bounding box of the "dog" in the image'
    img = np.array(Image.open(img_p))
    img = resize_image_short_side(img, 448)
    img_p = 'tmp_448.png'
    save_image(img, img_p)
    
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_p,
        device_map='auto',
        torch_dtype=torch.float16
    )
        
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_p, 
                                              min_pixels=min_pixels, 
                                              max_pixels=max_pixels)
    out = infer_single_img_qwen(model, processor, img_p, prompt, device)
    print("model answer:", out)



