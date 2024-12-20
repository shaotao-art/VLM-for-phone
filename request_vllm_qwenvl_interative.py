from openai import OpenAI
import base64
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import time
from colorama import Fore, Style, init

from utils.file_utils import read_image, get_image_base64
from utils.img_ops import smart_resize_img


def create_client(base_url, api_key):
    return OpenAI(base_url=base_url, 
                  api_key=api_key)

def get_model_response(client, model_path, messages, max_retries=3, delay=1):
    for _ in range(max_retries):
        try:
            # 开启流式传输
            completion = client.chat.completions.create(
                model=model_path,
                messages=messages,
                stream=True,
                timeout=10  # 设置超时
            )
            return completion
        except Exception as e:
            print(f"Error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return None



def main():
    init(autoreset=True)
    model_name = "qwen2_VL_7B"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    max_message_history = 10  # 最多保留10条历史记录
    img_tokens = 1024
    
    client = create_client(
        base_url="http://localhost:8001/v1",
        api_key="shaotao"
    )
    

    while True:
        text_inp = input(Fore.RED + "User 请输入text prompt: " + Style.RESET_ALL)
        if text_inp.lower() in ["q", "exit", "quit"]:
            print("Exiting the chat.")
            break
        img_path = input("请输入图像路径（如无图像输入直接回车）: ").strip()
        if img_path.strip() != '':
            img = read_image("data/elephant.jpg")
            img = smart_resize_img(img, img_tokens)
            user_inp = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": get_image_base64(img),
                        },
                    },
                    {"type": "text", "text": text_inp},
                ]
            }
        else:
            user_inp = {
            "role": "user",
                "content": [
                    {"type": "text", "text": text_inp},
                ]
            }
        messages.append(user_inp)
        if len(messages) > max_message_history:
            messages = messages[-max_message_history:]
        completion = get_model_response(client, model_name, messages)
        
        if completion:
            print(Fore.GREEN + "Model: ", end="", flush=True)
            model_resp = ""
            # 流式打印输出
            for chunk in completion:
                chunk_text = chunk.choices[0].delta.content
                print(chunk_text, end="", flush=True)
                model_resp += chunk_text
            print(Style.RESET_ALL)  # 重置颜色
            messages.append({"role": "assistant", "content": model_resp})
        else:
            print("对不起，模型无法响应。")

if __name__ == "__main__":
    main()
