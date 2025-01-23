#!/bin/bash

project_root=/home/shaotao/PROJECTS/VLM_AND_PHONE
ckp_root=/home/shaotao/PROJECTS/VLM_AND_PHONE/custom_lora_train/saves/01-23/original_text_with_aug
cuda_idx=0
model_type=2b
port=8003
test_img_tokens=1344
temprature=0.0
prompt_type=ground_prompt_test
eval_last=true

cd $ckp_root
all_ckp_folders=$(find . -type d -name 'checkpoint-*' | awk -F'-' '{print $2, $0}' | sort -n | cut -d' ' -f2)
if [[ ${eval_last} == true ]]; then
    all_ckp_folders=$(echo ${all_ckp_folders} | awk '{print $NF}')
fi


for ckp_folder in ${all_ckp_folders}
do
    cd $ckp_root
    ckp_path=$(realpath $ckp_folder)
    run_name=$(basename $(dirname $ckp_path))
    step_name=$(basename $ckp_path) 
    step_num=$(echo ${step_name} | cut -d'-' -f2)
    model_name=${run_name}-s-${step_num}
    echo '>>>'
    echo 'run name' $run_name
    echo 'step name' $step_name
    echo 'step num' $step_num
    echo 'model name' $model_name
    echo 'ckp path' $ckp_path
    echo 'serving and testing'

    # if [[ ${step_num} -lt 260 ]]; then
    #     echo 'skip...'
    #     continue
    # fi

    cd /home/shaotao/PROJECTS/VLM_AND_PHONE/scripts
    bash ./serve_qwenvl.sh -c ${cuda_idx} -p ${port} -m ${model_name} -k ${ckp_path} -t ${model_type} > ./tmp_${model_name}.log 2>&1 &
    # sleep 50 to wait for the server to start
    echo "sleep 120 s for server to start"
    sleep 120


    cd /home/shaotao/PROJECTS/VLM_AND_PHONE/eval/screenspot
    bash ./test_screen_spot.sh -m ${model_name} -p ${port} -t ${test_img_tokens} -r ${temprature} -a ${prompt_type}
    bash ./eval_sp.sh -m ${model_name} -p 'identity' -t ${test_img_tokens} -r ${temprature}

    echo "kill server"
    pid=$(fuser ${port}/tcp 2>/dev/null | awk '{print $1}' | sed 's#/tcp##')
    kill $pid
    kill $pid
    sleep 10
    echo "sleep 10s for server to stop"
done
