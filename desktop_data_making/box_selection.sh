iof_thres=0.5
img_root=/home/shaotao/DATA/os-altas/os-altas-macos
ann_root=/home/shaotao/DATA/os-altas/mac_ann
select_mode=random
sample_per_img=64

python box_selection.py \
    --img_root $img_root \
    --ann_root $ann_root \
    --iof_thres $iof_thres \
    --select_mode $select_mode \
    --sample_per_img $sample_per_img \
    --out_file_name 'mac_random_64.json'


select_mode=patch
num_segments=4
sample_per_patch=4

python box_selection.py \
    --img_root $img_root \
    --ann_root $ann_root \
    --iof_thres $iof_thres \
    --select_mode $select_mode \
    --num_segments $num_segments \
    --sample_per_patch $sample_per_patch \
    --out_file_name 'mac_patch_4_4.json'


iof_thres=0.5
img_root=/home/shaotao/DATA/os-altas/os-altas-linux
ann_root=/home/shaotao/DATA/os-altas/linux_ann
select_mode=random
sample_per_img=64

python box_selection.py \
    --img_root $img_root \
    --ann_root $ann_root \
    --iof_thres $iof_thres \
    --select_mode $select_mode \
    --sample_per_img $sample_per_img \
    --out_file_name 'linux_random_64.json'


select_mode=patch
num_segments=4
sample_per_patch=4

python box_selection.py \
    --img_root $img_root \
    --ann_root $ann_root \
    --iof_thres $iof_thres \
    --select_mode $select_mode \
    --num_segments $num_segments \
    --sample_per_patch $sample_per_patch \
    --out_file_name 'linux_patch_4_4.json'


