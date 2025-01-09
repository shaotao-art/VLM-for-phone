from typing import List, Tuple
from collections import defaultdict
import random

def patch_sampling(boxes: List[Tuple[float, float, float, float]], 
                    k: int, 
                    num_seg_h: int,
                    num_seg_w: int) -> List[int]:
    """Group boxes into segments and extract k boxes from each segment.
    
    Args:
    - boxes: List of boxes in format (x1, y1, x2, y2), in range [0, 1]
    - k: Number of boxes to extract from each segment
    - num_seg_h: Number of segments in height
    - num_seg_w: Number of segments in width
    
    Returns:
    - List of indices of extracted boxes
    """
    assert k > 0 and num_seg_h > 0 and num_seg_w > 0
    
    centers = [( (x1 + x2) / 2, (y1 + y2) / 2 ) for x1, y1, x2, y2 in boxes]
    groups = defaultdict(list)
    for idx, (cx, cy) in enumerate(centers):
        x_segment = int(cx * num_seg_w)
        y_segment = int(cy * num_seg_h)
        x_segment = min(num_seg_w - 1, x_segment)
        y_segment = min(num_seg_h - 1, y_segment)
        
        segment_key = (x_segment, y_segment)
        groups[segment_key].append(idx)
    
    extracted_indices = []
    for group in groups.values():
        random.shuffle(group)
        extracted_indices.extend(group[:k])
    
    random.shuffle(extracted_indices)
    return extracted_indices