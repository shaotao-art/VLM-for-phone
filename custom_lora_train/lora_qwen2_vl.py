"""
modified from https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune_distributed.py
"""
import torch
import json
import os
import argparse

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset
from functools import partial

from vision_utils import process_vision_info

from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator


    
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



    
class ToyDataSet(Dataset):
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)
   
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # use input_ids to avoid data be removed by huggingface
        # beacause huggingface trainer will remove keys not in model's forward function
        return dict(input_ids=self.data[idx])
    

def data_collate_fn(batch, processor, cut_off_len):   
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
    assert len(messages) == len(input_ids_lists)

    # get labels
    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list) # -100 is the ignore index in loss function
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    inputs['labels'] = labels_ids
    
    # print('text:', texts[0])
    # print('input_ids:', inputs['input_ids'][0].tolist())
    # print('labels:', inputs['labels'][0].tolist())
    # for b, c in zip(inputs['input_ids'][0].tolist(), inputs['labels'][0].tolist()):
    #     a = processor.tokenizer.decode(b)
    #     if a == '\n':
    #         a = 'newline'
    #     print(f'{a}, {b}, {c}')
    
    # import pdb; pdb.set_trace()

    
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

def get_lora_targets(model, freeze_vision_tower: bool):
    r"""
    Finds all available modules to apply lora or galore.
    copy from llamafactory
    """
    model_type = getattr(model.config, "model_type", None)
    assert model_type == 'qwen2_vl'
    forbidden_modules = {"lm_head"}
    forbidden_modules.add("merger")

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
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--model_path", type=str, required=True,
                      help="qwen2-vl pretrain model path")
    parser.add_argument("--lora_r", type=int, default=8,
                      help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                      help="LoRA alpha")
    parser.add_argument("--freeze_vision_tower", action="store_true", default=False,
                      help="whether to freeze vision tower parameters")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="number of workers for data loading")
    
    # train data args
    parser.add_argument("--train_data_path", type=str, required=True,
                      help="train data path")
    parser.add_argument("--min_img_tokens", type=int, default=256,
                      help="min img tokens")
    parser.add_argument("--max_img_tokens", type=int, required=True,
                      help="max img tokens")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--cut_off_len", type=int, required=True,
                      help="maximum sequence length")
    
    # TrainingArguments args
    parser.add_argument("--output_dir", type=str, required=True, help="output dir")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--bf16", action="store_true", default=True, help="use bfloat16 precision")
    parser.add_argument("--eval_strategy", type=str, default="no",choices=["no", "steps", "epoch"])
    parser.add_argument("--report_to", nargs="+", default=["tensorboard"], help="report tool list")
    parser.add_argument("--run_name", type=str, required=True, help="experiment run name")
    
    args = parser.parse_args()
    return args


def train():
    accelerator = Accelerator()
    args = get_args()
    assert args.cut_off_len > args.max_img_tokens, 'cut off len should be larger than the number of max img token'
    # print args in rank 0
    if accelerator.is_main_process:
        print("Training args:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        # save args
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "train_argparse.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    

    # load model
    model_path = args.model_path
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" # need to change in old gpus
    )
    
    # use lora to train, get peft model
    lora_target_modules = get_lora_targets(model, freeze_vision_tower=args.freeze_vision_tower)
    if accelerator.is_main_process:
        print(f"Traget lroa modules: {lora_target_modules}")
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_modules,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()


    # Load processor
    min_pixels = args.min_img_tokens*28*28
    max_pixels = args.max_img_tokens*28*28
    processor = AutoProcessor.from_pretrained(model_path, 
                                                min_pixels=min_pixels, 
                                                max_pixels=max_pixels, 
                                                padding_side="right")

    # get train dataset
    train_dataset = ToyDataSet(args.train_data_path)


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        eval_strategy=args.eval_strategy,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        run_name=args.run_name,
        dataloader_num_workers=args.num_workers,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=partial(data_collate_fn, processor=processor, cut_off_len=args.cut_off_len),
        
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if accelerator.is_main_process:
        write_chat_template(processor, args.output_dir)

if __name__ == "__main__":
    train()