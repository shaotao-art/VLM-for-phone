#!/bin/bash

while getopts p:m:t:i: flag
do
    case "${flag}" in
        p) port=${OPTARG};;
        m) model_name=${OPTARG};;
        t) max_img_tokens=${OPTARG};;
        i) inp_json_p=${OPTARG};;
    esac
done

api_key=shaotao
api_base=http://localhost:${port}/v1
img_root=/home/shaotao/DATA/aitw/aitw_images
out_json_p=bm-2b-aitw_m_${model_name}_itm_${max_img_tokens}.json
python request_vllm_qwenvl_mt_aitw.py \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --max_img_tokens ${max_img_tokens} \
    --out_json_p ${out_json_p} \
    --use_smart_resize \
    --img_root ${img_root} \
    --model_name ${model_name} \
    --inp_json_p ${inp_json_p} \


    
