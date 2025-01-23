#!/bin/bash

INPUT_FILE=/home/shaotao/PROJECTS/VLM_AND_PHONE/eval/aitw/out/01-19/albu-cot-thought-rep-fix-data-s-2366/bm-2b-aitw_m_albu-cot-thought-rep-fix-data-s-2366_itm_1344.json
python eval_aitw.py --data_path ${INPUT_FILE}