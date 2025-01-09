from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from peft import PeftModel
import torch
import argparse


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--peft_model_id", type=str, default=None)
    args.add_argument("--output_dir", type=str, default=None)
    args = args.parse_args()
    print("Merging the weights of the PEFT model with the base model")
    print(f"PEFT model id: {args.peft_model_id}")
    print(f"Output directory: {args.output_dir}")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained('/home/shaotao/PRETRAIN-CKPS/qwen2-vl-2b')
    processor = Qwen2VLProcessor.from_pretrained('/home/shaotao/PRETRAIN-CKPS/qwen2-vl-2b')
    model = PeftModel.from_pretrained(base_model, args.peft_model_id)
    model.merge_and_unload()
    # save the model
    model.base_model.model.to(torch.bfloat16).save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)