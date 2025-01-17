#!/bin/bash
project_root=/home/shaotao/PROJECTS/VLM_AND_PHONE
model_type='2b'
ckp_root=/home/shaotao/PROJECTS/VLM_AND_PHONE/custom_lora_train/custom_lora_saves/bm-2b-mind2web-itm-1280-lr-1e-4
cuda_idx=2
port=8002
test_img_tokens=1280

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

    if [[ ${step_num} -eq 250 || ${step_num} -eq 750 || ${step_num} -eq 500 ]]; then
        echo 'skip...'
        continue
    fi

    
    cd ${project_root}
    bash ./serve_qwenvl.sh -c ${cuda_idx} -p ${port} -m ${model_name} -k ${ckp_path} -t ${model_type} > tmp_${model_name}.log 2>&1 &
    # sleep 50 to wait for the server to start
    echo "sleep 50 s for server to start"
    sleep 50

    cd /home/shaotao/PROJECTS/VLM_AND_PHONE/eval/mind2web
    bash ./test_mind2web.sh -p ${port} -m ${model_name} -t ${test_img_tokens}
    bash ./eval_mind2web.sh -m ${model_name} -t ${test_img_tokens}

    echo "kill server"
    pid=$(fuser ${port}/tcp 2>/dev/null | awk '{print $1}' | sed 's#/tcp##')
    kill $pid
    kill $pid
    sleep 10
    echo "sleep 10s for server to stop"
done





