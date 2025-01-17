#!/bin/bash

project_root=/home/shaotao/PROJECTS/VLM_AND_PHONE
ckp_root=/home/shaotao/PROJECTS/VLM_AND_PHONE/custom_lora_train/custom_lora_saves/1-8/rep-4096-aug-float-final-rep-tune-head-beta2
cuda_idx=1
model_type='2b'
port=8007
test_img_tokens=1344
temprature=0.0

cd $ckp_root

all_ckp_folders=$(find . -type d -name 'checkpoint-*' | awk -F'-' '{print $2, $0}' | sort -n | cut -d' ' -f2)

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
    echo 'serving and testing'

    if [[ ${step_num} -lt 550 ]]; then
        echo 'skip...'
        continue
    fi

    cd /home/shaotao/PROJECTS/VLM_AND_PHONE/utils
    merge_weight='/home/shaotao/tmp_merge'
    python merge_weight.py --peft_model_id ${ckp_path} --output_dir ${merge_weight}
    

    cd ${project_root}
    echo $model_type
    echo $merge_weight
    pwd
    bash ./serve_qwenvl.sh -c ${cuda_idx} -p ${port} -m ${model_name} -k ${merge_weight} -t ${model_type} > tmp_${model_name}.log 2>&1 &
    # sleep 50 to wait for the server to start
    echo "sleep 50 s for server to start"
    sleep 60



    cd /home/shaotao/PROJECTS/VLM_AND_PHONE/eval/screenspot
    bash ./test_screen_spot.sh -m ${model_name} -p ${port} -t ${test_img_tokens} -r ${temprature}
    bash ./eval_sp.sh -m ${model_name} -p 'identity' -t ${test_img_tokens} -r ${temprature}

    echo "kill server"
    pid=$(fuser ${port}/tcp 2>/dev/null | awk '{print $1}' | sed 's#/tcp##')
    kill $pid
    kill $pid
    sleep 10
    echo "sleep 10s for server to stop"

done
