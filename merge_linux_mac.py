
from utils.file_utils import read_json, save_json
import pandas as pd


def make_data(json_file_p):
    data = pd.read_json(json_file_p)
    all_data = []
    for g_name, g_data in data.groupby("img_name"):
        one_img = dict()
        one_img["img_url"] = g_name
        element_lst = []
        for _, row in g_data.iterrows():
            # print(row['text'])
            box = row['box']
            pt = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            element_lst.append({
                "instruction": row['model_pred'],
                "bbox": box,
                "point": pt,
                "text": row['text']
            })
        one_img["element"] = element_lst
        all_data.append(one_img)
    return all_data


sample_type = 'random'
model_name = 'albu-box2func-only-som'
data1 = make_data(f"./scripts/out/mac_{sample_type}-{model_name}.json")
data2 = make_data(f"./scripts/out/linux_{sample_type}-{model_name}.json")
data = data1 + data2
print('data1 len:', len(data1))
print('data2 len:', len(data2))
print('data len:', len(data))
save_json(data, f"desktop-mada-{sample_type}-{model_name}.json")





