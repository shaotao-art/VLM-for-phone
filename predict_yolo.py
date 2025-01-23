"""used for detecting icons from images using yolo

two filter steps:
1. filter out boxes with large area, which are not icons
2. filter out boxes with width/height ratio not around 1
"""
from ultralytics import YOLO
import torch
import os

from utils.draw_utils import draw_box
from utils.file_utils import read_image, save_json


if __name__ == '__main__':
    model_path = '/home/shaotao/PRETRAIN-CKPS/Omniparser/icon_detect_v1_5/model_v1_5.pt'
    device = 'cuda'
    img_root = '/home/shaotao/DATA/os-altas/desktop-merged'
    out_json_path = 'desktop_merged_icons.json'
    max_ratio = 1.4 # icon width/height ratio should be around 1
    box_threshold = 0.5 # high value to keep only conf box
    iou_threshold = 0.2 # small value to avoid overlaped boxes, which should be presented in UI screenshots
    imgsz = 2560 # image large side size
    
    model = YOLO(model_path)
    model.to(device)

    file_lst = os.listdir(img_root)
    file_lst = [f for f in file_lst if 'sub' not in f]

    # make data
    all_data = []
    for img_idx in range(len(file_lst)):
    # for img_idx in range(200):
        image_path = os.path.join(img_root, file_lst[img_idx])
        try:
            ori_img = read_image(image_path)
        except:
            print('read image failed:', image_path)
            continue
        h, w = ori_img.shape[:2]
        result = model.predict(
            source=ori_img,
            imgsz=imgsz,
            conf=box_threshold,
            iou=iou_threshold, # default 0.7
            )
        boxes = result[0].boxes.xyxy #.tolist() # in pixel space
        conf = result[0].boxes.conf
        box_xyxy_01 = boxes / torch.Tensor([w, h, w, h]).to(boxes.device)
        box_xyxy_01 = box_xyxy_01.tolist()
        box_filtered = []
        element_lst = []
        for i in range(len(box_xyxy_01)):
            box = box_xyxy_01[i]
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > (1/20)**2:
                continue
            w_h_ratio = ((x2 - x1) * w) / ((y2 - y1) * h)
            if  1/max_ratio < w_h_ratio and w_h_ratio < max_ratio:
                box_filtered.append(box)
                element_lst.append(dict(
                    bbox=box,
                    # point=[(x1 + x2) / 2, (y1 + y2) / 2]
                ))        
        all_data.append(dict(
            img_url=os.path.basename(image_path),
            element=element_lst
        )) 
        
    save_json(all_data, out_json_path)
