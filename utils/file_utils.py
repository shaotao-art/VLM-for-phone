import json
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from typing import List, Dict
import pickle



def read_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_image(img: np.ndarray, 
               save_path: str):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)

def read_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def read_jsonl(filename):
    with open(filename) as f:
        data = [json.loads(line) for line in f]
    return data

def save_json(data: List[Dict], filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def get_image_base64(image_np: np.ndarray,
                     format: str='PNG'):

    pil_image = Image.fromarray(image_np)
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    encoded_image = base64.b64encode(img_byte_arr)
    encoded_image_text = encoded_image.decode('utf-8')
    base64_code = f"data:image/{format.lower()};base64,{encoded_image_text}"
    return base64_code

def read_pkl(pkl_path: str):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, pkl_path: str):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)