"""
process output of ominiparser

1. filter boxes based on IoF with a given threshold, keeping only the first of any overlapping boxes.
2. select boxes based on two modes: random or patch
3. save selected boxes to a json file
    data format:
    [
        {
            'img_name': 'xxx.png',
            'box': [x1, y1, x2, y2],
            'type': 'text' or 'icon',
            'text': 'xxx'
        },
        ...
    ]
"""
import os
import sys
from tqdm import tqdm
import random
import torch
from typing import List, Tuple
from collections import defaultdict

random.seed(42)

sys.path.append('/home/shaotao/PROJECTS/VLM_AND_PHONE')
from utils.file_utils import read_json, read_image, save_image, save_json
from utils.draw_utils import draw_box
from utils.helper_utils import print_args
import argparse




def group_and_extract_boxes(boxes: List[Tuple[float, float, float, float]], 
                            k: int, 
                            segments: int) -> List[int]:
    """Group boxes based on their centers and extract k boxes from each group"""
    centers = [( (x1 + x2) / 2, (y1 + y2) / 2 ) for x1, y1, x2, y2 in boxes]
    
    groups = defaultdict(list)
    for idx, (cx, cy) in enumerate(centers):
        x_segment = int(cx * segments)
        y_segment = int(cy * segments)
        x_segment = min(segments - 1, x_segment)
        y_segment = min(segments - 1, y_segment)
        
        segment_key = (x_segment, y_segment)
        groups[segment_key].append(idx)
    
    extracted_indices = []
    for group in groups.values():
        random.shuffle(group)
        extracted_indices.extend(group[:k])
    
    return extracted_indices


def calculate_iof_matrix(boxes: torch.Tensor) -> torch.Tensor:
    """Calculate the IoF matrix between boxes."""
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    boxes_expanded = boxes.unsqueeze(1)  # Shape: (N, 1, 4)
    
    inter_xmin = torch.max(boxes_expanded[:, :, 0], boxes_expanded[:, :, 0].transpose(0, 1))
    inter_ymin = torch.max(boxes_expanded[:, :, 1], boxes_expanded[:, :, 1].transpose(0, 1))
    inter_xmax = torch.min(boxes_expanded[:, :, 2], boxes_expanded[:, :, 2].transpose(0, 1))
    inter_ymax = torch.min(boxes_expanded[:, :, 3], boxes_expanded[:, :, 3].transpose(0, 1))
    
    inter_width = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_height = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_areas = inter_width * inter_height
    
    iof_matrix = inter_areas / areas.unsqueeze(1)
    return iof_matrix

def filter_boxes_by_iof_threshold(boxes, threshold):
    """Filter boxes based on IoF with a given threshold, keeping only the first of any overlapping boxes."""
    iof_matrix = calculate_iof_matrix(boxes)
    # Set diagonal to 0 since we do not want to consider self-overlap
    iof_matrix.fill_diagonal_(0)
    keep_mask = torch.ones(len(boxes), dtype=torch.bool)
    
    for i in range(len(boxes)):
        if not keep_mask[i]:
            continue
        high_iof_indices = (iof_matrix[i] >= threshold).nonzero(as_tuple=True)[0]
        # Mark all boxes with IoF >= threshold as not to be kept, except the first occurrence
        keep_mask[high_iof_indices] = False
        keep_mask[i] = True
    # Get indices of boxes that should be kept
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    return keep_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some images and annotations.')
    parser.add_argument('--img_root', type=str, help='Root directory of images')
    parser.add_argument('--ann_root', type=str, help='Root directory of annotations')
    parser.add_argument('--out_file_name', type=str, help='Output file name')
    parser.add_argument('--iof_thres', type=float, help='IoF threshold for filtering boxes')
    parser.add_argument('--show', action='store_true', help='Whether to show images with drawn boxes')
    parser.add_argument('--select_mode', type=str, choices=['random', 'patch'], help='Selection mode')
    parser.add_argument('--sample_per_img', type=int, help='Number of samples per image for random mode')
    parser.add_argument('--num_segments', type=int, help='Number of segments for patch mode')
    parser.add_argument('--sample_per_patch', type=int, help='Number of samples per patch for patch mode')

    args = parser.parse_args()
    print_args(args)

    img_root = args.img_root
    ann_root = args.ann_root
    out_file_name = args.out_file_name
    iof_thres = args.iof_thres
    show = args.show
    select_mode = args.select_mode
    sample_per_img = args.sample_per_img
    num_segments = args.num_segments
    sample_per_patch = args.sample_per_patch
    assert select_mode in ['random', 'patch']
    
    
    num_ele_lst = []
    all_data = []
    # only use whole img
    img_list = list(sorted(_ for _ in os.listdir(img_root) if 'sub' not in _))
    for idx in tqdm(range(len(img_list))):
        img_name = img_list[idx]
        img_path = os.path.join(img_root, img_name)
        if not os.path.exists(img_path):
            print('one file do not exsist')
            continue
        
        ann_path = os.path.join(ann_root, img_name.replace('.png', '.json'))
        if show:
            img = read_image(img_path)
            
        try:
            ann = read_json(ann_path)
        except:
            print('one error, skipping...')
            continue

        # box are preordered, text first, icon second
        all_boxes = torch.tensor([line['bbox'] for line in ann])
        if select_mode == 'patch':
            keep_idx = filter_boxes_by_iof_threshold(all_boxes, iof_thres)
            patch_keep_idx = group_and_extract_boxes(all_boxes[keep_idx], sample_per_patch, num_segments)
            final_keep_idx = keep_idx[patch_keep_idx].tolist()
            random.shuffle(final_keep_idx)
        else:
            keep_idx = filter_boxes_by_iof_threshold(all_boxes, iof_thres)
            random.shuffle(keep_idx.tolist())
            final_keep_idx = keep_idx[:sample_per_img]
        
        num_ele_lst.append(len(final_keep_idx))
        
        for b_idx, line in enumerate(ann):
            if b_idx not in final_keep_idx:
                continue
            type_ = line['type']
            box = line['bbox']
            text = line['content']
            is_interactable = line['interactivity']
            x1, y1, x2, y2 = box
            b_w, b_h = x2 - x1, y2 - y1
            box_area = b_w * b_h
            # some filters
            if b_w / b_h > 10 or b_h / b_w > 3:
                continue
            if box_area > 1 * 1 * 0.05:
                continue
            if type_ == 'text' and (text is None or len(text) <= 3 or len(text) >= 30):
                continue

            if show:
                img = draw_box(img, box)

            all_data.append(dict(
                img_name=img_name,
                box=box,
                type=type_,
                text=text
            ))
        if show:
            save_image(img, f'tmp_{idx}.png')
        # break
    print('data num:', len(all_data))
    print('mean num of elements:', sum(num_ele_lst) / len(num_ele_lst))
    save_json(all_data, out_file_name)


