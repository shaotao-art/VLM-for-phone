#!/bin/bash
MODEL_NAME='amex-5k-mtfunc'
PASER_METHOD='kuohao'
TEST_IMG_TOKENS=1280

##--------------before run----------------##
##      CHANGE PROMPT IN VAL SCRIPT       ##
##--------------before run----------------## 

INPUT_FILE=./out/sp_mobile_m_${MODEL_NAME}_itm_${TEST_IMG_TOKENS}.json
python screenspot_val.py --res_file_p ${INPUT_FILE} \
    --method ${PASER_METHOD}

INPUT_FILE=./out/sp_desktop_m_${MODEL_NAME}_itm_${TEST_IMG_TOKENS}.json
python screenspot_val.py --res_file_p ${INPUT_FILE} \
    --method ${PASER_METHOD}

INPUT_FILE=./out/sp_web_m_${MODEL_NAME}_itm_${TEST_IMG_TOKENS}.json
python screenspot_val.py --res_file_p ${INPUT_FILE} \
    --method ${PASER_METHOD}
