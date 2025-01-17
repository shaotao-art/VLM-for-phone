#!/bin/bash
img_root='/home/shaotao/DATA/aitw/aitw_images'
hist_len=4
cot_ann_p='/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/aitw-l3.xlsx'
train_json_p='/home/shaotao/DATA/aitw/aitw_data_train.json'

# ## train data making
# python aitw.py --img_root ${img_root} \
#     --inp_json_p ${train_json_p} \
#     --out_json_p '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/aitw_train_naive.json' \
#     --hist_len ${hist_len}

# python aitw.py --img_root ${img_root} \
#     --inp_json_p ${train_json_p} \
#     --out_json_p '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/aitw_train_action.json' \
#     --hist_len ${hist_len} \
#     --cot_ann_p ${cot_ann_p} \
#     --cot_level 'action'

# python aitw.py --img_root ${img_root} \
#     --inp_json_p ${train_json_p} \
#     --out_json_p '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/aitw_train_thought.json' \
#     --hist_len ${hist_len} \
#     --cot_ann_p ${cot_ann_p} \
#     --cot_level 'thoughts'


test_json_p='/home/shaotao/DATA/aitw/aitw_data_test.json'
## train data making
python aitw.py --img_root ${img_root} \
    --inp_json_p ${test_json_p} \
    --out_json_p '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/aitw_test_naive.json' \
    --hist_len ${hist_len}

python aitw.py --img_root ${img_root} \
    --inp_json_p ${test_json_p} \
    --out_json_p '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/aitw_test_action.json' \
    --hist_len ${hist_len} \
    --cot_ann_p ${cot_ann_p} \
    --cot_level 'action'

python aitw.py --img_root ${img_root} \
    --inp_json_p ${test_json_p} \
    --out_json_p '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/aitw_test_thought.json' \
    --hist_len ${hist_len} \
    --cot_ann_p ${cot_ann_p} \
    --cot_level 'thoughts'