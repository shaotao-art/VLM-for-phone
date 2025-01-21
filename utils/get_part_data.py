import json
import random

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)



if __name__ == '__main__':
    input_file_path = '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/new_format/amex_data.json'
    output_file_path = '/home/shaotao/PROJECTS/VLM_AND_PHONE/baseline_data/new_format/amex_data_5k.json'
    data_num = 5000
    data = read_json(input_file_path)
    random.shuffle(data)
    processed_data = data[:data_num]
    write_json(output_file_path, processed_data)
    print(f"Done! Processed {len(data)} items into {len(processed_data)} items.")