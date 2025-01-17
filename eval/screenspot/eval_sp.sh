#!/bin/bash
# Parse arguments
while getopts m:p:t:r: flag
do
    case "${flag}" in
        m) model_name=${OPTARG};;
        p) parse_method=${OPTARG};;
        t) max_img_tokens=${OPTARG};;
        r) temprature=${OPTARG};;
    esac
done

formatted_date=$(date +"%m-%d")
inp_json_p=./out/${formatted_date}/${model_name}/sp_mobile_m_${model_name}_itm_${max_img_tokens}-t_${temprature}.json
python screenspot_val.py --res_file_p ${inp_json_p} \
    --method ${parse_method}

inp_json_p=./out/${formatted_date}/${model_name}/sp_desktop_m_${model_name}_itm_${max_img_tokens}-t_${temprature}.json
python screenspot_val.py --res_file_p ${inp_json_p} \
    --method ${parse_method}

inp_json_p=./out/${formatted_date}/${model_name}/sp_web_m_${model_name}_itm_${max_img_tokens}-t_${temprature}.json
python screenspot_val.py --res_file_p ${inp_json_p} \
    --method ${parse_method}
