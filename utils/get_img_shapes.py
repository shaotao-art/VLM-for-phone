import os
import sys
PROJECT_ROOT='/home/shaotao/PROJECTS/VLM_AND_PHONE'
sys.path.append(PROJECT_ROOT)
from glob import glob
from tqdm import tqdm
import os
import pickle 
import logging

from utils.file_utils import read_image
from concurrent.futures import ThreadPoolExecutor


logging.basicConfig(level=logging.INFO)


def process_file(file):
    try:
        if 'omniact' in file:
            filename = '/'.join(file.split('/')[-2:])
        else:
            filename = os.path.basename(file)

        img = read_image(file)
        assert filename not in img_name2shape
        return filename, img.shape[:2]
    except Exception as e:
        logging.info(f'one error in {file}')
        logging.error(e)
        return None

if __name__ == '__main__':
    img_files = glob('/home/shaotao/DATA/mind2web/ming2web_images/*')
    out_pkl_p = 'mind2web_img_shapes.pkl'
    img_name2shape = dict()

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(process_file, img_files), total=len(img_files)))

    for result in results:
        if result is not None:
            filename, shape = result
            img_name2shape[filename] = shape

    # save to pickle
    with open(out_pkl_p, 'wb') as f:
        pickle.dump(img_name2shape, f)