import torch
import json
import os
import random
import sys
sys.path.append('/home/shaotao/PROJECTS/VLM_AND_PHONE')
import argparse
import numpy as np
from PIL import Image
import yaml
from functools import partial
from typing import List

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info


from prompts import all_prompts
from utils.file_utils import save_image
from utils.helper_utils import get_date_str
from custom_dataset import GroundingDataset, Loc2FuncDataset, AgentDataset

random.seed(42)


    
def find_assistant_content_sublist_indexes(l):
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, 
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    start_indexes = []
    end_indexes = []

    for i in range(len(l) - 1):
        if l[i] == 151644 and l[i + 1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2)
                    break
    return list(zip(start_indexes, end_indexes))

        

def data_collate_fn(batch, cut_off_len, processor):
    if 'conv_lst' in batch[0]['input_ids']:
        # for grounding dataset
        texts = [processor.apply_chat_template(msg['input_ids']['conv_lst'], 
                                            tokenize=False, 
                                            add_generation_prompt=False) for msg in batch]
        imgs = [msg['input_ids']['img'] for msg in batch]
        inputs = processor(
            text=texts, 
            images=imgs,
            padding=True, 
            return_tensors="pt"
        )
    else:
        # for agent dataset
        messages = [m['input_ids']['messages'] for m in batch]
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    
    input_ids_lists = inputs['input_ids'].tolist()

    # get labels
    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list) # -100 is the ignore index in loss function
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    inputs['labels'] = labels_ids
    if labels_ids.shape[1] > cut_off_len:
        inputs['input_ids'] = inputs['input_ids'][:, :cut_off_len]
        inputs['attention_mask'] = inputs['attention_mask'][:, :cut_off_len] 
        inputs['labels'] = inputs['labels'][:, :cut_off_len]
    return inputs
    

def write_chat_template(processor, output_dir):
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n"
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)

def get_lora_targets(model, freeze_vision_tower: bool, freeze_lm_head: bool):
    r"""
    Finds all available modules to apply lora or galore.
    copy from llamafactory
    """
    model_type = getattr(model.config, "model_type", None)
    assert model_type == 'qwen2_vl'
    forbidden_modules = set()
    forbidden_modules.add("merger")
    
    if freeze_lm_head:
        forbidden_modules.add("lm_head")

    if freeze_vision_tower:
        forbidden_modules.add("visual")

    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])
    module_names = list(module_names)
    
    if freeze_vision_tower:
        return "^(?!.*visual).*(?:{}).*".format("|".join(module_names))
    else:
        # exclude conv3d, which is not supported by peft
        return "^(?!.*patch_embed).*(?:{}).*".format("|".join(module_names))



def get_args():
    parser = argparse.ArgumentParser(description="Load config from file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)


def train():
    accelerator = Accelerator()
    args = get_args()
    assert args.cut_off_len > args.max_img_tokens, 'cut off len should be larger than the number of max img token'

    date_str = get_date_str()                    
    output_dir = os.path.join('saves', date_str, args.run_name)

    n_gpus = 0
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    # print args in rank 0
    if accelerator.is_main_process:
        print("Training args:")
        print(f"  n_gpus: {n_gpus}")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        # save args
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train_argparse.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    

    # qlora config
    bnb_config = None
    if args.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # bnb_4bit_quant_storage=torch.bfloat16,
        )
        

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype='auto',
        attn_implementation=args.attn_impl,
        quantization_config=bnb_config,
        device_map=args.device_map
    )
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    
    if args.use_lora:
        # use lora to train, get peft model
        lora_target_modules = get_lora_targets(model, 
                                               freeze_vision_tower=args.freeze_vision_tower,
                                               freeze_lm_head=args.freeze_lm_head)
        if accelerator.is_main_process:
            print(f"Traget lroa modules: {lora_target_modules}")
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    
    # Gradient checkpointing
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()


    # Load processor
    min_pixels = args.min_img_tokens*28*28
    max_pixels = args.max_img_tokens*28*28
    processor = AutoProcessor.from_pretrained(args.model_path, 
                                              min_pixels=min_pixels, 
                                              max_pixels=max_pixels, 
                                              padding_side="right")

    # get train dataset
    if args.dataset_type == 'grounding':
        train_dataset = GroundingDataset(data_path=args.train_data_path,
                                        img_root=args.img_root,
                                        init_prompt=all_prompts[args.init_prompt],
                                        pt_format=args.pt_format,
                                        ele_per_img=args.ele_per_img,
                                        crop_min=args.crop_min,
                                        crop_max=args.crop_max
                                        )
    elif args.dataset_type == 'loc2func':
        train_dataset = Loc2FuncDataset(data_path=args.train_data_path,
                                        img_root=args.img_root,
                                        init_prompt=all_prompts[args.init_prompt],
                                        pt_format=args.pt_format,
                                        ele_per_img=args.ele_per_img,
                                        use_som=args.use_som,
                                        use_ocr=args.use_ocr,
                                        crop_min=args.crop_min,
                                        crop_max=args.crop_max
                                        )
    elif args.dataset_type == 'agent':
        train_dataset = AgentDataset(data_path=args.train_data_path,
                                     img_root=args.img_root)
    else:
        raise ValueError(f"dataset type {args.dataset_type} is not supported")
    
    
    if accelerator.is_main_process:
        print(f"train dataset size: {len(train_dataset)}")
        print('sample data:')
        sample = train_dataset[2]['input_ids']
        if args.dataset_type == 'agent':
            print(processor.apply_chat_template(sample['messages'], tokenize=False, add_generation_prompt=False))
        elif args.dataset_type == 'grounding':
            print(processor.apply_chat_template(sample['conv_lst'], tokenize=False, add_generation_prompt=False))
            img = np.array(sample['img'])
            save_image(img, f'sample_img_{args.run_name}.jpg')
            gpts = sample['conv_lst'][1::2]
            from utils.draw_utils import draw_dot
            for i in range(len(gpts)):
                pt = eval(gpts[i]['content'][0]['text'])
                if pt[0] > 1 or pt[1] > 1:
                    pt[0], pt[1] = pt[0]/1000, pt[1]/1000
                img = draw_dot(img, pt)
            save_image(img, f'sample_img_{args.run_name}.jpg')
            
            
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        eval_strategy=args.eval_strategy,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        run_name=args.run_name,
        dataloader_num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters
    )
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=partial(data_collate_fn,
                              cut_off_len=args.cut_off_len,
                              processor=processor),
        
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if accelerator.is_main_process:
        write_chat_template(processor, output_dir)

if __name__ == "__main__":
    train()