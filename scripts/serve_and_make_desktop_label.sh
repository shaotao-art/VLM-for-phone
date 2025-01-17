#/bin/bash
# device_idx=3
# port=8007
# ckp_path=/home/shaotao/PROJECTS/VLM_AND_PHONE/custom_lora_train/saves/01-16/albu-box2-func-null/checkpoint-2520
# model_type=2b
# prompt_type=box2func_test
# model_name=albu-box2func-naive
# vis=


device_idx=3
port=8008
ckp_path=/home/shaotao/PROJECTS/VLM_AND_PHONE/custom_lora_train/saves/01-17/albu-box2-func-ocr-som/checkpoint-2520
model_type=2b
prompt_type=box2func_with_ocr_and_som_test
model_name=albu-box2func-ocr-som
vis=--vis


bash ./serve_qwenvl.sh -c ${device_idx} -p ${port} -m ${model_name} -k ${ckp_path} -t ${model_type} > ../logs/box2func-${model_name}.log  2>&1 &
# sleep 50 to wait for the server to start
echo "sleep 60 s for server to start"
sleep 60

openai_api_key=shaotao
openai_api_base=http://localhost:${port}/v1
temprature=0.0
num_threads=20
max_img_tokens=1344
use_smart_resize='--use_smart_resize'



img_root=/home/shaotao/DATA/os-altas/os-altas-macos
inp_json_p=/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/data_making_source_data/mac_patch_4_4.json
out_json_p=mac_patch-${model_name}.json
python ../request_vllm_qwenvl_mt_desktop.py \
    --openai_api_key ${openai_api_key} \
    --openai_api_base ${openai_api_base} \
    --temprature ${temprature} \
    --num_thread ${num_threads} \
    --model_name ${model_name} \
    --prompt_type ${prompt_type} \
    ${use_smart_resize} \
    --inp_json_p ${inp_json_p} \
    --out_json_p ${out_json_p} \
    --img_root ${img_root} \
    --max_img_tokens ${max_img_tokens} \
    ${vis}


img_root=/home/shaotao/DATA/os-altas/os-altas-macos
inp_json_p=/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/data_making_source_data/mac_random_64.json
out_json_p=mac_random-${model_name}.json
python ../request_vllm_qwenvl_mt_desktop.py \
    --openai_api_key ${openai_api_key} \
    --openai_api_base ${openai_api_base} \
    --temprature ${temprature} \
    --num_thread ${num_threads} \
    --model_name ${model_name} \
    --prompt_type ${prompt_type} \
    ${use_smart_resize} \
    --inp_json_p ${inp_json_p} \
    --out_json_p ${out_json_p} \
    --img_root ${img_root} \
    --max_img_tokens ${max_img_tokens} \
    ${vis}


img_root=/home/shaotao/DATA/os-altas/os-altas-linux
inp_json_p=/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/data_making_source_data/linux_patch_4_4.json
out_json_p=linux_patch-${model_name}.json
python ../request_vllm_qwenvl_mt_desktop.py \
    --openai_api_key ${openai_api_key} \
    --openai_api_base ${openai_api_base} \
    --temprature ${temprature} \
    --num_thread ${num_threads} \
    --model_name ${model_name} \
    --prompt_type ${prompt_type} \
    ${use_smart_resize} \
    --inp_json_p ${inp_json_p} \
    --out_json_p ${out_json_p} \
    --img_root ${img_root} \
    --max_img_tokens ${max_img_tokens} \
    ${vis}


img_root=/home/shaotao/DATA/os-altas/os-altas-linux
inp_json_p=/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/data_making_source_data/linux_random_64.json
out_json_p=linux_random-${model_name}.json
python ../request_vllm_qwenvl_mt_desktop.py \
    --openai_api_key ${openai_api_key} \
    --openai_api_base ${openai_api_base} \
    --temprature ${temprature} \
    --num_thread ${num_threads} \
    --model_name ${model_name} \
    --prompt_type ${prompt_type} \
    ${use_smart_resize} \
    --inp_json_p ${inp_json_p} \
    --out_json_p ${out_json_p} \
    --img_root ${img_root} \
    --max_img_tokens ${max_img_tokens} \
    ${vis}



echo "kill server"
pid=$(fuser ${port}/tcp 2>/dev/null | awk '{print $1}' | sed 's#/tcp##')
kill $pid
kill $pid
sleep 10
echo "sleep 10s for server to stop"