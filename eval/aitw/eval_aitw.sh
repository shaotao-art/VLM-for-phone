#!/bin/bash

INPUT_FILE=/home/shaotao/PROJECTS/VLM_AND_PHONE/eval/aitw/out/bm-2b-aitw_m_bm-2b-aitw-itm-1280-lr-1e-4-s-2366_itm_1280.json
python eval_aitw.py --data_path ${INPUT_FILE}