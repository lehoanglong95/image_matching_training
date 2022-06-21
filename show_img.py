import cv2
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

def on_press(event):
    if event.key == "x":
        plt.close("all")

def to_list(val):
    return [val]

if __name__ == '__main__':
    error_indices_1_neighbor = np.load("./output/error_indices_1.npy")[0]
    df = pd.read_parquet("./input_file/final_product_image_with_label_test_set.parquet")[["normalized_url_image", "label"]]
    df = df.reset_index()
    del df["index"]
    df = df.loc[error_indices_1_neighbor]
    df["normalized_url_image"] = df["normalized_url_image"].apply(to_list)
    print(len(df))
    new_df = df.groupby("label").agg({"normalized_url_image": "sum"})
    print(len(new_df))
    base_dir = "/Users/lap02387/pms/image_matching/images"
    for idx, row in new_df.iterrows():
        if len(row["normalized_url_image"]) < 2:
            continue
        fig, ax = plt.subplots(1, 2)
        fig.canvas.mpl_connect("key_press_event", on_press)
        first_file_name = row["normalized_url_image"][0].split("/")[-1]
        second_file_name = row["normalized_url_image"][1].split("/")[-1]
        img_1 = io.imread(f"{base_dir}/{first_file_name}")
        img_2 = io.imread(f"{base_dir}/{second_file_name}")
        ax[0].imshow(img_1)
        ax[1].imshow(img_2)
        plt.show()


