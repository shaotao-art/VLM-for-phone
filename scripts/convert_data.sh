cd ../utils
INPUT_P=/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/action_caption/aitw-action-caption-action.json
OUTPUT_P=/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/action_caption/aitw-action-caption-action-converted.json
python convert_llamafac_to_custom.py --input  $INPUT_P\
    --output $OUTPUT_P