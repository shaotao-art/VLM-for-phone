import json
from math import ceil
import random

random.seed(42)

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def process_elements(data, element_per_list):
    all_data = []
    for item in data:
        elements = item.get('element', [])
        random.shuffle(elements)
        num_new_lists = ceil(len(elements) / element_per_list)
        for i in range(num_new_lists):
            start_index = i * element_per_list
            end_index = start_index + element_per_list
            chunk = elements[start_index:end_index]
            if len(chunk) > 0:
                all_data.append({
                    "img_url": item["img_url"],
                    "element": chunk
                })
    return all_data

if __name__ == '__main__':
    input_file_path = '/home/shaotao/PROJECTS/VLM_AND_PHONE/final_ablu_data/desktop_ocr_elements.json'
    output_file_path = input_file_path.replace('.json', '-split.json')
    ele_per_list = 5
    data = read_json(input_file_path)
    processed_data = process_elements(data, element_per_list=ele_per_list)
    write_json(output_file_path, processed_data)
    print(f"Done! Processed {len(data)} items into {len(processed_data)} items.")