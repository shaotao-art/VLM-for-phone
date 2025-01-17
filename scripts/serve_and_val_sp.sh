#!/bin/bash

# args for MODEL_NAME and CKP_P
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_name> <checkpoint_path>"
    exit 1
fi
MODEL_NAME=$1
CKP_P=$2

export CUDA_VISIBLE_DEVICES=3
PORT=8003
API_KEY="shaotao"
API_BASE="http://localhost:${PORT}/v1"
IMG_ROOT='/home/shaotao/DATA/screen_spot/images'
MAX_IMG_TOKENS=1280
PROMPT_TYPE="ground_prompt"

### EVAL
PASER_METHOD='identity'
TEST_IMG_TOKENS=1280

## 2B MODEL
echo "starting server"
model_p=/home/shaotao/PRETRAIN-CKPS/qwen2-vl-2b
vllm serve $model_p --dtype float16 \
                    --port ${PORT} \
                    --api-key ${API_KEY} \
                    --served-model-name qwen2_VL_2B \
                    --max-model-len 1550 \
                    --limit-mm-per-prompt "image=1" \
                    --enable-lora \
                    --lora-modules "${MODEL_NAME}=${CKP_P}"\
                    --enforce-eager  \
                    --disable-log-requests > output.log 2>&1 &
pid=$!
echo "server pid: $pid"

# sleep 15 to wait for the server to start
echo "sleep 50 s for server to start"
sleep 50


# start eval
echo "start eval"
cd /home/shaotao/PROJECTS/VLM_AND_PHONE/eval/screenspot
INP_JSON_P='/home/shaotao/DATA/screen_spot/screenspot_web.json'
OUT_JSON_P=sp_web_m_${MODEL_NAME}_itm_${MAX_IMG_TOKENS}.json
python request_vllm_qwenvl_mt_screenspot.py --model_name "$MODEL_NAME" \
        --inp_json_p "$INP_JSON_P" \
        --out_json_p "$OUT_JSON_P" \
        --img_root "$IMG_ROOT" \
        --max_img_tokens "$MAX_IMG_TOKENS" \
        --prompt_type $PROMPT_TYPE \
        --use_smart_resize \
        --openai_api_key $API_KEY \
        --openai_api_base $API_BASE 


INP_JSON_P='/home/shaotao/DATA/screen_spot/screenspot_mobile.json'
OUT_JSON_P=sp_mobile_m_${MODEL_NAME}_itm_${MAX_IMG_TOKENS}.json
python request_vllm_qwenvl_mt_screenspot.py --model_name "$MODEL_NAME" \
        --inp_json_p "$INP_JSON_P" \
        --out_json_p "$OUT_JSON_P" \
        --img_root "$IMG_ROOT" \
        --max_img_tokens "$MAX_IMG_TOKENS" \
        --prompt_type $PROMPT_TYPE \
        --use_smart_resize \
        --openai_api_key $API_KEY \
        --openai_api_base $API_BASE 


INP_JSON_P='/home/shaotao/DATA/screen_spot/screenspot_desktop.json'
OUT_JSON_P=sp_desktop_m_${MODEL_NAME}_itm_${MAX_IMG_TOKENS}.json
python request_vllm_qwenvl_mt_screenspot.py --model_name "$MODEL_NAME" \
        --inp_json_p "$INP_JSON_P" \
        --out_json_p "$OUT_JSON_P" \
        --img_root "$IMG_ROOT" \
        --max_img_tokens "$MAX_IMG_TOKENS" \
        --prompt_type $PROMPT_TYPE \
        --use_smart_resize \
        --openai_api_key $API_KEY \
        --openai_api_base $API_BASE 


echo "start cal accuracy"


# eval
cd /home/shaotao/PROJECTS/VLM_AND_PHONE/eval/screenspot
INPUT_FILE=./out/sp_mobile_m_${MODEL_NAME}_itm_${TEST_IMG_TOKENS}.json
python screenspot_val.py --res_file_p ${INPUT_FILE} \
    --method ${PASER_METHOD}

INPUT_FILE=./out/sp_desktop_m_${MODEL_NAME}_itm_${TEST_IMG_TOKENS}.json
python screenspot_val.py --res_file_p ${INPUT_FILE} \
    --method ${PASER_METHOD}

INPUT_FILE=./out/sp_web_m_${MODEL_NAME}_itm_${TEST_IMG_TOKENS}.json
python screenspot_val.py --res_file_p ${INPUT_FILE} \
    --method ${PASER_METHOD}

echo "kill server"
kill $pid 2>/dev/null
kill $pid 2>/dev/null
echo "sleep 10s for server to stop"
sleep 10
