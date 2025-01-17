#!/bin/bash
img_root='/home/shaotao/DATA/mind2web/ming2web_images'
img_shape_pkl_p='/home/shaotao/PROJECTS/VLM_AND_PHONE/train_val_data_making/agent_task/mind2web_img_shapes.pkl'
hist_len=4
cot_ann_p='/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/mind2web-l3.xlsx'

## train data making
python mind2web.py --img_root ${img_root} \
    --inp_json_p '/home/shaotao/DATA/mind2web/mind2web_data_train.json' \
    --out_json_p '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/mind2web_train_naive.json' \
    --img_shape_pkl_p ${img_shape_pkl_p} \
    --hist_len ${hist_len}

python mind2web.py --img_root ${img_root} \
    --inp_json_p '/home/shaotao/DATA/mind2web/mind2web_data_train.json' \
    --out_json_p '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/mind2web_train_action.json' \
    --img_shape_pkl_p ${img_shape_pkl_p} \
    --hist_len ${hist_len} \
    --cot_ann_p ${cot_ann_p} \
    --cot_level 'action'

python mind2web.py --img_root ${img_root} \
    --inp_json_p '/home/shaotao/DATA/mind2web/mind2web_data_train.json' \
    --out_json_p '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/mind2web_train_thought.json' \
    --img_shape_pkl_p ${img_shape_pkl_p} \
    --hist_len ${hist_len} \
    --cot_ann_p ${cot_ann_p} \
    --cot_level 'thoughts'


## test data making
python mind2web.py --img_root ${img_root} \
    --inp_json_p '/home/shaotao/DATA/mind2web/mind2web_data_test_task.json' \
    --out_json_p '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/mind2web_test_task.json' \
    --img_shape_pkl_p ${img_shape_pkl_p} \
    --hist_len ${hist_len}

python mind2web.py --img_root ${img_root} \
    --inp_json_p '/home/shaotao/DATA/mind2web/mind2web_data_test_domain.json' \
    --out_json_p '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/mind2web_test_domain.json' \
    --img_shape_pkl_p ${img_shape_pkl_p} \
    --hist_len ${hist_len}

python mind2web.py --img_root ${img_root} \
    --inp_json_p '/home/shaotao/DATA/mind2web/mind2web_data_test_website.json' \
    --out_json_p '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/mind2web_test_website.json' \
    --img_shape_pkl_p ${img_shape_pkl_p} \
    --hist_len ${hist_len}
