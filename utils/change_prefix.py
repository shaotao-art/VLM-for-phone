import os
import sys
PROJECT_ROOT=os.path.dirname(os.path.abspath('.'))
sys.path.append(PROJECT_ROOT)
import argparse

from utils.file_utils import read_json, save_json
from utils.helper_utils import print_args


## for llamafactory data
def replace_img_prefix_llamafactory(json_file_path, old_prefix, new_prefix, output_file_path):
    data = read_json(json_file_path)
    for item in data:
        if 'image_lst' in item:
            new_img_lst = []
            for file_path in item['image_lst']:
                if  old_prefix in file_path:
                    file_path = file_path.replace(old_prefix, new_prefix)
                    new_img_lst.append(file_path)
                else:
                    print('Error: file path does not contain the old prefix')
            item['image_lst'] = new_img_lst
    save_json(data, output_file_path)
        


## for my own data
def replace_image_prefix_custom(json_file_path, old_prefix, new_prefix, output_file_path):
    json_data = read_json(json_file_path)
    for line in json_data:
        for message in line.get('messages', []):
            for content_item in message.get('content', []):
                if content_item.get('type') == 'image':
                    image_path = content_item['image']
                    content_item['image'] = image_path.replace(old_prefix, new_prefix)
    save_json(json_data, output_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replace image prefixes in JSON files.')
    parser.add_argument('--old_prefix', type=str, required=True, help='Old prefix to be replaced')
    parser.add_argument('--new_prefix', type=str, required=True, help='New prefix to replace the old one')
    parser.add_argument('--inp_p', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--out_p', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--method', type=str, required=True, choices=['llamafactory', 'custom'], help='Method to use for prefix replacement')

    args = parser.parse_args()
    print_args(args)

    old_prefix = args.old_prefix
    new_prefix = args.new_prefix
    inp_p = args.inp_p
    out_p = args.out_p
    method = args.method
    if method == 'llamafactory':
        replace_img_prefix_llamafactory(inp_p, old_prefix, new_prefix, out_p)
    elif method == 'custom':
        replace_image_prefix_custom(inp_p, old_prefix, new_prefix, out_p)
    else:
        raise ValueError(f"Unknown method: {method}")
    print("Image prefix replacement completed.")