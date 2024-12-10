import os
import sys
PROJECT_ROOT=os.path.dirname(os.path.abspath('.'))
sys.path.append(PROJECT_ROOT)
from glob import glob
from tqdm import tqdm
import os
import pickle 
import logging

from utils.file_utils import read_image


logging.basicConfig(level=logging.INFO)



if __name__ == '__main__':
    img_files = glob('/home/shaotao/DATA/omniact-SHOWUI-8k/screenshots/*/*')
    out_pkl_p = 'omniact_img_shapes.pkl'
    img_name2shape = dict()
    for file in tqdm(img_files):
        try:
            if 'omniact' in file:
                filename = '/'.join(file.split('/')[-2:])
            else:
                filename = os.path.basename(file)

            img = read_image(file)
            assert filename not in img_name2shape
            img_name2shape[filename] = img.shape[:2]
            
        except Exception as e:
            logging.info(f'one error in {file}')
            logging.error(e)
        # break

    # save to pickle
    with open(out_pkl_p, 'wb') as f:
        pickle.dump(img_name2shape, f)