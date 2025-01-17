#/bin/bash

# Parse arguments
while getopts p:m:t: flag
do
    case "${flag}" in
        p) port=${OPTARG};;
        m) model_name=${OPTARG};;
        t) max_img_tokens=${OPTARG};;
    esac
done


openai_api_key=shaotao
openai_api_base=http://localhost:${port}/v1
temprature=0.0
num_threads=20
model_name=${model_name}
prompt_type=box2func_test
use_smart_resize='--use_smart_resize'
inp_json_p=/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/data_making_source_data/mac_patch_4_4.json
out_json_p='mac_patch_train-som.json'
img_root=/home/shaotao/DATA/os-altas/os-altas-macos
vis=

cd ..
python request_vllm_qwenvl_mt_desktop.py \
    --openai_api_key ${openai_api_key} \
    --openai_api_base ${openai_api_base} \
    --temprature ${temprature} \
    --num_thread ${num_threads} \
    --model_name ${model_name} \
    --prompt_type ${prompt_type} \
    --use_smart_resize ${use_smart_resize} \
    --inp_json_p ${inp_json_p} \
    --out_json_p ${out_json_p} \
    --img_root ${img_root} \
    --max_img_tokens ${max_img_tokens} \
    ${vis}