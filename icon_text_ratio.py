
import pandas as pd
import random
from matplotlib import pyplot as plt

random.seed(32)


def print_stat(data):
    data['num_icon'] = data['element_icon'].apply(len)
    data['num_text'] = data['element_text'].apply(len)
    general_icon_text_ratio = data['num_icon'].sum() / data['num_text'].sum()
    print('general icon text ratio: ', general_icon_text_ratio)
    # plt.hist(data['icon_text_ratio'], bins=20, range=(0, 3))
    # plt.show()

def sample_elements(df, icon_text_ratio):
    sampled_df = df.copy()
    for index, row in sampled_df.iterrows():
        num_icon = len(row['element_icon'])
        num_text = len(row['element_text'])
        
        if icon_text_ratio < 1:
            sample_size = int(num_text * icon_text_ratio)
            sampled_icons = random.sample(row['element_icon'], min(sample_size, num_icon))
            sampled_df.at[index, 'element_icon'] = sampled_icons
        else:
            sample_size = int(num_icon / icon_text_ratio)
            sampled_texts = random.sample(row['element_text'], min(sample_size, num_text))
            sampled_df.at[index, 'element_text'] = sampled_texts
    
    return sampled_df


def merge_icon_text(df):
    df['element'] = df.apply(lambda x: x['element_icon'] + x['element_text'], axis=1)
    df = df[['img_url', 'element']]
    return df

if __name__ == '__main__':
    icon_path = '/home/shaotao/PROJECTS/VLM_AND_PHONE/final_ablu_data/data_making/box2func-omniact-desktop_icon_elements.json'
    text_path = '/home/shaotao/PROJECTS/VLM_AND_PHONE/final_ablu_data/data_making/box2func-omniact-desktop_ocr_elements.json'
    out_path = '/home/shaotao/PROJECTS/VLM_AND_PHONE/final_ablu_data/data_making/box2func-omniact-training.json'
    icon_text_ratio = 3/2
    
    data1 = pd.read_json(icon_path)
    data2 = pd.read_json(text_path)
    data = pd.merge(data1, data2, on='img_url', how='inner', suffixes=('_icon', '_text'))
    print_stat(data)
    sampled_df = sample_elements(data, icon_text_ratio)
    print_stat(sampled_df)
    sampled_df = merge_icon_text(sampled_df)
    sampled_df.to_json(out_path, orient='records')
    sampled_df['num_elements'] = sampled_df['element'].apply(len)
    print('mean num elements: ', sampled_df['num_elements'].mean())






