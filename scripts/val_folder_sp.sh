#!/bin/bash

project_root=/home/shaotao/PROJECTS/VLM_AND_PHONE
ckp_root=/home/shaotao/PROJECTS/VLM_AND_PHONE/custom_lora_train/saves/01-09/guiact_single_baseline_aug_3x
cuda_idx=2
model_type='2b'
port=8008
test_img_tokens=1344
temprature=0.0
prompt_type='ground_prompt'
img_first='-f'

cd $ckp_root

all_ckp_folders=$(find . -type d -name 'checkpoint-*' | awk -F'-' '{print $2, $0}' | sort -nr | cut -d' ' -f2)

# all_ckp_folders=$(find . -type d -name "checkpoint-*")
# print all ckp folders one by one
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

    if [[ ${step_num} -lt 2100 ]]; then
        echo 'skip...'
        continue
    fi

    cd ${project_root}
    bash ./serve_qwenvl.sh -c ${cuda_idx} -p ${port} -m ${model_name} -k ${ckp_path} -t ${model_type} > ./logs/tmp_${model_name}.log 2>&1 &
    # sleep 50 to wait for the server to start
    echo "sleep 60 s for server to start"
    sleep 60


    cd /home/shaotao/PROJECTS/VLM_AND_PHONE/eval/screenspot
    bash ./test_screen_spot.sh -m ${model_name} -p ${port} -t ${test_img_tokens} -r ${temprature} -a ${prompt_type} ${img_first}
    bash ./eval_sp.sh -m ${model_name} -p 'identity' -t ${test_img_tokens} -r ${temprature}

    echo "kill server"
    pid=$(fuser ${port}/tcp 2>/dev/null | awk '{print $1}' | sed 's#/tcp##')
    kill $pid
    kill $pid
    sleep 10
    echo "sleep 10s for server to stop"
done
