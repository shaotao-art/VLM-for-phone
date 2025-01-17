#!/bin/bash
img_root='/home/shaotao/DATA/aitw/aitw_images'
hist_len=1000
cot_ann_p='/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/aitw-l3.xlsx'
train_json_p='/home/shaotao/DATA/aitw/aitw_data_train.json'
test_json_p='/home/shaotao/DATA/aitw/aitw_data_test.json'
use_cot_his=

# python aitw_action_caption.py --img_root ${img_root} \
#     --inp_json_p ${train_json_p} \
#     --out_json_p 'aitw-action-caption-naive.json' \
#     --hist_len ${hist_len} \
#     --cot_ann_p ${cot_ann_p} \
#     --cot_level 'action' \
#     ${use_cot_his}

python aitw_action_caption_test.py --img_root ${img_root} \
    --inp_json_p ${test_json_p} \
    --out_json_p 'aitw-test-action-making.json' \
    --hist_len ${hist_len}