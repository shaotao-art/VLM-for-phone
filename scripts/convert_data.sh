cd ../utils
INPUT_P=/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/aitw_train_good_hist.json
OUTPUT_P=/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/agent_data/aitw_train_good_hist_converted.json
python convert_llamafac_to_custom.py \
    --input  $INPUT_P\
    --output $OUTPUT_P