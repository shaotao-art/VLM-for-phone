## 7B model
# export CUDA_VISIBLE_DEVICES=3
# model_p=/home/shaotao/PRETRAIN-CKPS/qwen2-vl-7b
# vllm serve $model_p --dtype float16 \
#                     --port 8001 \
#                     --api-key shaotao \
#                     --served-model-name qwen2_VL_7B \
#                     --enable-lora \
#                     --lora-modules "bm-7b-amex-7b-fs-s-320-=/home/shaotao/PROJECTS/VLM_AND_PHONE/training/custom_lora_saves/bm-7b-amex-fs-7b-itm-512-lr-1e-4/checkpoint-320" \
#                     --enforce-eager \
#                     --max-model-len 1500 
#                     # --tensor-parallel-size 2 \


## 2B MODEL
export CUDA_VISIBLE_DEVICES=0
model_p=/home/shaotao/PRETRAIN-CKPS/qwen2-vl-2b
vllm serve $model_p --dtype float16 \
                    --port 8001 \
                    --api-key shaotao \
                    --served-model-name qwen2_VL_2B \
                    --max-model-len 1550 \
                    --limit-mm-per-prompt "image=1" \
                    --enable-lora \
                    --lora-modules "amex-5k-mtfunc=/home/shaotao/PROJECTS/VLM_AND_PHONE/training/custom_lora_saves/bm-2b-amex-5k-mtfunc-itm-1280-lr-1e-4/checkpoint-624"\
                    --enforce-eager  \
                    --disable-log-requests