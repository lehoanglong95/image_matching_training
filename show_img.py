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
    # df = pd.read_parquet("./input_file/final_product_image_with_label_test_set.parquet")[["normalized_url_image", "label"]]
    df = pd.read_parquet("./output/similarity_df_3.parquet")
    df = df.reset_index()
    del df["index"]
    del df["duplicated"]
    # df = df.loc[error_indices_1_neighbor]
    # df["normalized_url_image"] = df["normalized_url_image"].apply(to_list)
    # print(len(df))
    # new_df = df.groupby("label").agg({"normalized_url_image": "sum"})
    # print(len(new_df))
    base_dir = "/Users/lap02387/pms/image_matching/images"
    for idx, row in df.iterrows():
        if len(row["normalized_url_image"]) < 2:
            continue
        fig, ax = plt.subplots(1, 5)
        fig.canvas.mpl_connect("key_press_event", on_press)
        label = row["normalized_url_image"]
        candidate_1 = row["duplicated_1"]
        candidate_2 = row["duplicated_2"]
        candidate_3 = row["duplicated_3"]
        duplicated = row["correct_image"]
        img_1 = io.imread(f"{base_dir}/{label}")
        img_2 = io.imread(f"{base_dir}/{candidate_1}")
        img_3 = io.imread(f"{base_dir}/{candidate_2}")
        img_4 = io.imread(f"{base_dir}/{candidate_3}")
        img_5 = io.imread(f"{base_dir}/{duplicated}")
        ax[0].imshow(img_1)
        ax[1].imshow(img_2)
        ax[2].imshow(img_3)
        ax[3].imshow(img_4)
        ax[4].imshow(img_5)
        plt.show()


