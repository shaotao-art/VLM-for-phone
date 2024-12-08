import argparse

from utils.file_utils import read_json, save_json
from utils.helper_utils import print_args

def transform_sharegpt_to_custom(original_data):
    out_data = []
    for line in original_data:
        transformed = dict(messages=[])
        image_lst = line.get("image_lst")
        conversation_lst = line.get("conversation")
        
        img_idx = 0
        for turn in conversation_lst:
            turn_role = turn['from']
            assert turn_role in ['human', 'gpt']
            turn_value = turn['value']
            if turn_role == 'human':
                if '<image>' in turn_value:
                    transformed['messages'].append({
                            'role': 'user',
                            'content': [
                                {'type': 'image', 'image': image_lst[img_idx]},
                                {'type': 'text', 'text': turn_value.replace('<image>\n', '')}
                            ]
                        })
                    img_idx += 1
                else:
                    transformed['messages'].append({
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': turn_value}
                            ]
                        })
            else:
                transformed['messages'].append(
                        {
                            'role': 'assistant',
                            'content': [{'type': 'text', 'text': turn_value}]
                        })
            
        try:
            assert img_idx == len(image_lst)
            out_data.append(transformed)
        except:
            print('error diag: ', line)
    return out_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ShareGPT data to custom format.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output JSON file.')
    
    args = parser.parse_args()
    print_args(args)
    
    data_p = args.input
    out_data_p = args.output
    original_data = read_json(data_p)
    print('original data_num: ', len(original_data))
    transformed_data = transform_sharegpt_to_custom(original_data)
    print('final data_num: ', len(transformed_data))
    save_json(transformed_data, out_data_p)
