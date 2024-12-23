#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

#---------common----------#
# 模型相关参数
MODEL_PATH="/home/shaotao/PRETRAIN-CKPS/qwen2-vl-2b"
LORA_R=8
LORA_ALPHA=16

# 训练数据相关参数
MIN_IMG_TOKENS=4

# 训练参数
PER_DEVICE_TRAIN_BATCH_SIZE=1
NUM_TRAIN_EPOCHS=2.0
SAVE_TOTAL_LIMIT=5
SAVE_STEPS=320
MAX_GRAD_NORM=1.0
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=1e-4
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.1
LOGGING_STEPS=5
BF16=true
EVAL_STRATEGY="no"
# REPORT_TO="wandb"
REPORT_TO="tensorboard"
#---------common----------#




#---------run----------#
TRAIN_DATA_PATH="/home/shaotao/PROJECTS/VLM_AND_PHONE/custom_train_data/amex/train/amex_5k_func_custom_train.json"
MAX_IMG_TOKENS=1280
CUT_OFF_LEN=1670
RUN_NAME=bm-2b-amex-5k-mtfunc-itm-${MAX_IMG_TOKENS}-lr-1e-4
OUTPUT_DIR=custom_lora_saves/${RUN_NAME}


python lora_qwen2_vl.py \
    --model_path ${MODEL_PATH} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --train_data_path ${TRAIN_DATA_PATH} \
    --min_img_tokens ${MIN_IMG_TOKENS} \
    --max_img_tokens ${MAX_IMG_TOKENS} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --save_steps ${SAVE_STEPS} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --logging_steps ${LOGGING_STEPS} \
    --eval_strategy ${EVAL_STRATEGY} \
    --report_to ${REPORT_TO} \
    --run_name ${RUN_NAME} \
    --bf16 \
    --cut_off_len ${CUT_OFF_LEN} \
    --freeze_vision_tower 