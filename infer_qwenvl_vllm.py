from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image
from qwen_vl_utils import process_vision_info


def infer_single_image(img_path, 
                       prompt, 
                       llm, 
                       sampling_params):
    # 创建用户消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # 设置图像像素范围
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    
    # 初始化处理器并应用聊天模板
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # 处理多模态信息
    image_inputs, video_inputs = process_vision_info(messages)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    
    # 构建 LLM 输入数据
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }
    
    # 生成输出
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    return generated_text

def infer_single_image_multi_turn(img_path, 
                       prompt, 
                       llm, 
                       sampling_params):
    img_path2 = '截屏2024-10-26 14.47.36.png'
    # 创建用户消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "如果你想安装《Legacy of Discord》，你应该点击屏幕上的“Install”按钮"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这个按钮具体在屏幕的哪个位置？"}
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "这个按钮位于页面中间偏上位置。"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "帮我再简要描述一下这张图片。"},
                {"type": "image", "image": img_path2}
            ],
        }
    ]
    
    # 设置图像像素范围
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    
    # 初始化处理器并应用聊天模板
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # 处理多模态信息
    image_inputs, video_inputs = process_vision_info(messages)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    
    # 构建 LLM 输入数据
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }
    print(llm_inputs)
    
    # 生成输出
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    return generated_text

# 主函数
if __name__ == "__main__":
    # 模型路径
    MODEL_PATH = "/home/dmt/shao-tao-working-dir/BIG_PROJ_LIN/models/qwen2_VL_2B"
    
    # 初始化语言模型 (LLM)
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 10, "video": 10},
        dtype='float16'
    )
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=256,
        stop_token_ids=[],
    )
    out = infer_single_image_multi_turn(
        img_path="./46.jpg",
        prompt="如果我要安装legacy of discord，我应该点击屏幕的哪个位置？",
        llm=llm,
        sampling_params=sampling_params
    )
    print(out)
    

