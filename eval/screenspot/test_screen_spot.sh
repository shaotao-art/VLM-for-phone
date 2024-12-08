API_KEY="shaotao"
API_BASE="http://localhost:8001/v1"
MODEL_NAME='amex-5k-mtfunc'
IMG_ROOT='/home/shaotao/DATA/screen_spot/images'
MAX_IMG_TOKENS=1280


INP_JSON_P='/home/shaotao/DATA/screen_spot/screenspot_web.json'
OUT_JSON_P=sp_web_m_${MODEL_NAME}_itm_${MAX_IMG_TOKENS}.json
python request_vllm_qwenvl_mt_screenspot.py --model_name "$MODEL_NAME" \
        --inp_json_p "$INP_JSON_P" \
        --out_json_p "$OUT_JSON_P" \
        --img_root "$IMG_ROOT" \
        --max_img_tokens "$MAX_IMG_TOKENS" \
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
        --use_smart_resize \
        --openai_api_key $API_KEY \
        --openai_api_base $API_BASE 
