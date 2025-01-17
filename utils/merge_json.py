import os
import sys
PROJECT_ROOT=os.path.dirname(os.path.abspath('.'))
sys.path.append(PROJECT_ROOT)
import random
from utils.file_utils import read_json, save_json

if __name__ == '__main__':
    file_lst = ['/home/shaotao/PROJECTS/VLM_AND_PHONE/data/12-30-box2func/amex-box2func-10k-ocr.json', 
                '/home/shaotao/PROJECTS/VLM_AND_PHONE/data/12-30-box2func/guiact-box2func.json']
    data = []
    for file in file_lst:
        part = read_json(file)
        print('Reading file: ', file) 
        print('Length: ', len(part))
        data.extend(part)
    
    random.shuffle(data)
    save_json(data, 'amex-guiact-box2func-ocr.json')
    