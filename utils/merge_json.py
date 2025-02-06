import os
import sys
PROJECT_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import random
from utils.file_utils import read_json, save_json

if __name__ == '__main__':
    file_lst = ['/home/shaotao/PROJECTS/VLM_AND_PHONE/final_ablu_data/grounding_amex.json', 
                '/home/shaotao/PROJECTS/VLM_AND_PHONE/final_ablu_data/grounding_guiact-single.json',
                '/home/shaotao/PROJECTS/VLM_AND_PHONE/final_ablu_data/grounding_omniact-split.json']
    data = []
    for file in file_lst:
        part = read_json(file)
        print('Reading file: ', file) 
        print('Length: ', len(part))
        data.extend(part)
    
    random.shuffle(data)
    save_json(data, 'grouding_amex-guiact-omniact-mix.json')
    