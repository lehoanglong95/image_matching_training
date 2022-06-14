import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from skimage import io

def read_image(input_path):
    try:
        image = io.imread(os.path.join("/data/long.le3/image_matching/images", input_path.split("/")[-1]))
        if image.shape[2] != 3:
            return {input_path: False}
        return {input_path: True}
    except Exception as ee:
        print(ee)
        return {input_path: False}
if __name__ == '__main__':
    df = pd.read_parquet("./input_file/final_final_product_image.parquet")
    normalized_source_image = df.normalized_url_image.values
    with Pool(8) as p:
        results = [x for x in tqdm(p.imap(read_image, normalized_source_image), total=len(normalized_source_image))]
    keys = []
    values = []
    for e in results:
        for k, v in e.items():
            keys.append(k)
            values.append(v)
    dict_results = dict(zip(keys, values))
    errors = [k for k, v in dict_results.items() if not v]
    print(len(errors))
    print(errors)
    # final_df = df[df["normalized_url_image"].isin(set(errors))]
    # final_df.to_parquet("./input_file/final_final_final_product_image.parquet")