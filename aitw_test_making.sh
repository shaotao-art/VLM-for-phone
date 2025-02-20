#!/bin/bash

while getopts p:m:t: flag
do
    case "${flag}" in
        p) port=${OPTARG};;
        m) model_name=${OPTARG};;
        t) max_img_tokens=${OPTARG};;
    esac
done

api_key=shaotao
api_base=http://localhost:${port}/v1
img_root=/home/shaotao/DATA/aitw/aitw_images

inp_json_p=/home/shaotao/PROJECTS/VLM_AND_PHONE/aitw_test_making.json
out_json_p=aitw_test_making_${model_name}.json
python request_vllm_qwenvl_mt_action_caption.py --api_key ${api_key} \
    --api_base ${api_base} \
    --max_img_tokens ${max_img_tokens} \
    --out_json_p ${out_json_p} \
    --use_smart_resize \
    --model_name ${model_name} \
    --inp_json_p ${inp_json_p} \
    --img_root ${img_root}


    
