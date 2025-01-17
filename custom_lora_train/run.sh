#!/bin/bash
config_path=./albu-desktop-data-making/random_som.yml
CUDA_VISIBLE_DEVICES=1 python lora_qwen2_vl.py --config ${config_path}

config_path=./albu-desktop-data-making/patch_som.yml
CUDA_VISIBLE_DEVICES=1 python lora_qwen2_vl.py --config ${config_path}

config_path=./albu-desktop-data-making/random_naive.yml
CUDA_VISIBLE_DEVICES=1 python lora_qwen2_vl.py --config ${config_path}

config_path=./albu-desktop-data-making/patch_naive.yml
CUDA_VISIBLE_DEVICES=1 python lora_qwen2_vl.py --config ${config_path}

# multi gpu
# accelerate launch --config_file='./configs/ddp.yaml' --main_process_port 29501 lora_qwen2_vl.py \
#         --config './sp-baseline-albu/guiact_single_baseline_aug_3x_2gpu.yml'