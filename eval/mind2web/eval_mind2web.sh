#!/bin/bash
# Parse arguments
while getopts m:t: flag
do
    case "${flag}" in
        m) model_name=${OPTARG};;
        t) max_img_tokens=${OPTARG};;
    esac
done

inp_json_p=./out/bm-2b-mind2web_m_${model_name}_itm_${max_img_tokens}-test_task.json
python cal_mind2web_acc.py --data_path ${inp_json_p}

inp_json_p=./out/bm-2b-mind2web_m_${model_name}_itm_${max_img_tokens}-test_website.json
python cal_mind2web_acc.py --data_path ${inp_json_p}

inp_json_p=./out/bm-2b-mind2web_m_${model_name}_itm_${max_img_tokens}-test_domain.json
python cal_mind2web_acc.py --data_path ${inp_json_p}