#/bin/bash

# Parse arguments
while getopts a:fp:m:t:r: flag
do
    case "${flag}" in
        a) prompt_type=${OPTARG};;
        f) img_first="--image_first";;  # Set image first flag if -f is passed
        p) port=${OPTARG};;
        m) model_name=${OPTARG};;
        t) max_img_tokens=${OPTARG};;
        r) temprature=${OPTARG};;
    esac
done


api_key=shaotao
api_base=http://localhost:${port}/v1
img_root=/home/shaotao/DATA/screen_spot/images


inp_json_p=/home/shaotao/DATA/screen_spot/screenspot_web.json
out_json_p=sp_web_m_${model_name}_itm_${max_img_tokens}-t_${temprature}.json
python request_vllm_qwenvl_mt_screenspot.py --model_name ${model_name} \
        --inp_json_p ${inp_json_p} \
        --out_json_p ${out_json_p} \
        --img_root ${img_root} \
        --max_img_tokens ${max_img_tokens} \
        --use_smart_resize \
        --prompt_type ${prompt_type} \
        --openai_api_key ${api_key} \
        --openai_api_base ${api_base} \
        --temprature ${temprature} \
        ${img_first}


inp_json_p='/home/shaotao/DATA/screen_spot/screenspot_mobile.json'
out_json_p=sp_mobile_m_${model_name}_itm_${max_img_tokens}-t_${temprature}.json
python request_vllm_qwenvl_mt_screenspot.py --model_name ${model_name} \
        --inp_json_p ${inp_json_p} \
        --out_json_p ${out_json_p} \
        --img_root ${img_root} \
        --max_img_tokens ${max_img_tokens} \
        --use_smart_resize \
        --prompt_type ${prompt_type} \
        --openai_api_key ${api_key} \
        --openai_api_base ${api_base} \
        --temprature ${temprature} \
        ${img_first}

inp_json_p='/home/shaotao/DATA/screen_spot/screenspot_desktop.json'
out_json_p=sp_desktop_m_${model_name}_itm_${max_img_tokens}-t_${temprature}.json
python request_vllm_qwenvl_mt_screenspot.py --model_name ${model_name} \
        --inp_json_p ${inp_json_p} \
        --out_json_p ${out_json_p} \
        --img_root ${img_root} \
        --max_img_tokens ${max_img_tokens} \
        --use_smart_resize \
        --prompt_type ${prompt_type} \
        --openai_api_key ${api_key} \
        --openai_api_base ${api_base} \
        --temprature ${temprature} \
        ${img_first}
