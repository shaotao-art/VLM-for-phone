cd ../utils
DATA_P=/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/aitw_train_action_converted.json
IMG_SHAPE_PKL_P='xx'
MAX_IMG_TOKENS=1344
DATA_FORMAT='custom'
MODEL_P='/home/shaotao/PRETRAIN-CKPS/qwen2-vl-2b'


python cal_data_len.py --data_p ${DATA_P} \
    --img_shape_pkl_p ${IMG_SHAPE_PKL_P} \
    --max_img_tokens ${MAX_IMG_TOKENS} \
    --data_format ${DATA_FORMAT} \
    --model_p ${MODEL_P}