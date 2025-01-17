#!/bin/bash

# Parse arguments
while getopts "c:p:m:k:t:l:" opt; do
    case ${opt} in
        c )
            cuda_idx=$OPTARG
            ;;
        p )
            port=$OPTARG
            ;;
        m )
            model_name=$OPTARG
            ;;
        k )
            ckp_path=$OPTARG
            ;;
        t )
            model_type=$OPTARG
            ;;
        l )
            max_seq_len=$OPTARG
            ;;
        \? )
            echo "Usage: cmd [-c cuda_idx] [-p port] [-m model_name] [-k ckp_path] [-t model_type] -l max_seq_len"
            exit 1
            ;;
    esac
done

# if max_seq_len is not provided, set it to 1550
max_seq_len=${max_seq_len:-1550}

export CUDA_VISIBLE_DEVICES=${cuda_idx}
if [[ ${model_type} == '2b' ]]; then
    model_p=/home/shaotao/PRETRAIN-CKPS/qwen2-vl-2b
    vllm serve ${model_p} --dtype bfloat16 \
                        --port ${port} \
                        --api-key shaotao \
                        --served-model-name 'qwen2-vl-2b' \
                        --max-model-len ${max_seq_len} \
                        --limit-mm-per-prompt "image=1" \
                        --enable-lora \
                        --lora-modules "${model_name}=${ckp_path}"
                        # --enforce-eager
                        # --disable-log-requests
                        

fi

if [[ ${model_type} == '7b' ]]; then
    # 7B model
    model_p=/home/shaotao/PRETRAIN-CKPS/qwen2-vl-7b
    vllm serve $model_p --dtype bfloat16 \
                        --port ${port} \
                        --api-key shaotao \
                        --served-model-name qwen2-vl-7b \
                        --max-model-len ${max_seq_len} \
                        --limit-mm-per-prompt "image=1" \
                        --enable-lora \
                        --lora-modules "${model_name}=${ckp_path}"
                        # --enforce-eager  \
                        # --disable-log-requests \
                        # --tensor-parallel-size 2
fi